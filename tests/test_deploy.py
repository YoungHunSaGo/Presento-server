"""배포 파일 점검(실제 AWS·Docker 호출 없음).

운영 compose, Caddy, IAM·CORS JSON, 스크립트, .env.prod.example.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"


@pytest.fixture(scope="module")
def prod():
    return yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text())


def test_prod_services(prod):
    assert set(prod["services"]) == {"api", "worker", "postgres", "redis", "caddy"}  # MinIO 없음
    assert all(s["restart"] == "unless-stopped" for s in prod["services"].values())


def test_only_caddy_is_exposed(prod):
    for name, svc in prod["services"].items():
        if name == "caddy":
            assert sorted(svc["ports"]) == ["443:443", "443:443/udp", "80:80"]
        else:
            assert "ports" not in svc, name  # postgres·redis·api는 외부에 열지 않는다


def test_worker_concurrency_one(prod):
    cmd = prod["services"]["worker"]["command"]
    assert cmd[cmd.index("--concurrency") + 1] == "1"


def test_app_uses_aws_s3_not_minio(prod):
    for svc in prod["services"].values():
        assert "minio" not in str(svc).lower()
    for name in ("api", "worker"):
        env = prod["services"][name]["environment"]
        assert "S3_ENDPOINT_URL" not in env  # 비어 있음 → AWS 기본 주소
        assert prod["services"][name]["env_file"] == ".env.prod"


def test_no_secret_defaults_in_compose():
    text = (ROOT / "docker-compose.prod.yml").read_text()
    # 비밀번호는 .env.prod에서만(없으면 compose가 실패)
    assert "${POSTGRES_PASSWORD:?" in text
    assert not re.search(r"POSTGRES_PASSWORD:\s*\w", text.replace("${POSTGRES_PASSWORD", ""))


def test_caddyfile_proxies_api_and_docs():
    text = (DEPLOY / "Caddyfile").read_text()
    assert text.count("{$DOMAIN}") == 1
    assert re.search(r"@api path .*?/api/v1/\*.*?/docs.*?/openapi\.json", text)
    assert "reverse_proxy api:8000" in text


def test_iam_policy_is_bucket_scoped():
    policy = json.loads((DEPLOY / "iam-policy.json").read_text())
    actions = {a for st in policy["Statement"] for a in st["Action"]}
    # HeadObject는 s3:GetObject 권한으로 허용된다(별도 액션 없음)
    assert actions == {"s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"}
    resources = {st["Resource"] for st in policy["Statement"]}
    assert resources == {"arn:aws:s3:::YOUR_BUCKET_NAME", "arn:aws:s3:::YOUR_BUCKET_NAME/*"}
    assert all(st["Effect"] == "Allow" for st in policy["Statement"])


def test_s3_cors_allows_frontend_put_get():
    (rule,) = json.loads((DEPLOY / "s3-cors.json").read_text())["CORSRules"]
    assert "http://localhost:5173" in rule["AllowedOrigins"]
    assert "*" not in rule["AllowedOrigins"]
    assert {"PUT", "GET"} <= set(rule["AllowedMethods"])


def test_env_prod_example_has_no_secrets():
    values = {}
    for line in (ROOT / ".env.prod.example").read_text().splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.partition("=")
            values[k] = v
    for secret in ("OPENAI_API_KEY", "JWT_SECRET", "POSTGRES_PASSWORD", "S3_ACCESS_KEY",
                   "S3_SECRET_KEY", "DOMAIN"):  # fmt: skip
        assert values[secret] == "", secret
    assert values["CORS_ORIGINS"].split(",")[0] == "http://localhost:5173"
    assert values["S3_REGION"] == "ap-northeast-2"


def test_env_prod_is_gitignored_but_example_is_not():
    if shutil.which("git") is None:
        pytest.skip("git 없음")

    def ignored(name):
        res = subprocess.run(  # noqa: S603
            ["git", "check-ignore", "-q", name],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
        )
        return res.returncode == 0

    assert ignored(".env.prod") and ignored(".env")
    assert not ignored(".env.prod.example")


@pytest.mark.parametrize("script", ["setup.sh", "update.sh", "swap.sh"])
def test_scripts_are_valid_bash(script):
    path = DEPLOY / script
    text = path.read_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    subprocess.run(["bash", "-n", str(path)], check=True)  # noqa: S603, S607


def test_update_script_migrates_before_restart():
    text = (DEPLOY / "update.sh").read_text()
    pull, build = text.index("git pull --ff-only"), text.index("dc build")
    migrate, up = text.index("alembic upgrade head"), text.index("dc up -d --remove-orphans")
    assert pull < build < migrate < up


def test_prod_project_name_is_separate_from_dev(prod):
    assert prod["name"] == "presento-prod"


def test_worker_memory_cap_and_graceful_stop(prod):
    worker = prod["services"]["worker"]
    assert worker["mem_limit"] == "1200m"
    # 진행 중 분석(최대 16분)이 끝날 때까지 기다린다
    assert worker["stop_grace_period"] == "17m"


def test_update_script_checks_s3_and_reexecs_after_pull():
    text = (DEPLOY / "update.sh").read_text()
    assert "head('deploy-check/none') is None" in text
    assert text.index("git pull --ff-only") < text.index('exec bash "$SELF" --no-pull')


def test_setup_waits_for_dpkg_lock():
    assert "DPkg::Lock::Timeout" in (DEPLOY / "setup.sh").read_text()
