"""집 서버 배포 파일 점검(서버·Docker 호출 없음): docker-compose.home.yml, .env.home.example,
deploy/home-update.sh."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.home.yml"
SCRIPT = ROOT / "deploy" / "home-update.sh"

MEM_LIMITS = {
    "api": "768m",
    "worker": "1500m",
    "postgres": "512m",
    "redis": "128m",
    "minio": "512m",
}
LONG_RUNNING = set(MEM_LIMITS)


@pytest.fixture(scope="module")
def home():
    return yaml.safe_load(COMPOSE.read_text())


@pytest.fixture(scope="module")
def env_example():
    values = {}
    for line in (ROOT / ".env.home.example").read_text().splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.partition("=")
            values[k] = v
    return values


def test_home_services(home):
    assert home["name"] == "presento-home"
    assert set(home["services"]) == LONG_RUNNING | {"minio-init"}


def test_restart_policy(home):
    for name in LONG_RUNNING:
        assert home["services"][name]["restart"] == "unless-stopped", name
    # 버킷 생성은 한 번 실행하고 끝나는 작업: unless-stopped면 성공해도 계속 다시 뜬다.
    # 실패했을 때만 다시 시도한다.
    assert home["services"]["minio-init"]["restart"].startswith("on-failure")


def test_log_rotation(home):
    for name, svc in home["services"].items():
        opts = svc["logging"]["options"]
        assert svc["logging"]["driver"] == "json-file", name
        assert (opts["max-size"], opts["max-file"]) == ("10m", "3"), name


def test_memory_limits(home):
    for name, limit in MEM_LIMITS.items():
        assert home["services"][name]["mem_limit"] == limit, name


def test_no_gpu_reservation(home):
    text = COMPOSE.read_text()
    assert "nvidia" not in text.lower() and "gpus" not in text
    for svc in home["services"].values():
        assert "deploy" not in svc and "runtime" not in svc and "devices" not in svc


def test_worker_concurrency_one_and_graceful_stop(home):
    worker = home["services"]["worker"]
    cmd = worker["command"]
    assert cmd[cmd.index("--concurrency") + 1] == "1"
    assert worker["stop_grace_period"] == "17m"  # 진행 중 분석(최대 16분)을 끊지 않는다


def test_ports_bound_to_bind_addr_and_configurable(home):
    s = home["services"]
    assert s["api"]["ports"] == ["${BIND_ADDR:-127.0.0.1}:${API_PORT:-8000}:8000"]
    assert s["minio"]["ports"] == ["${BIND_ADDR:-127.0.0.1}:${MINIO_PORT:-9000}:9000"]
    for name in ("postgres", "redis", "worker", "minio-init"):
        assert "ports" not in s[name], name  # 호스트에 열지 않는다


def test_s3_internal_and_public_endpoints(home):
    for name in ("api", "worker"):
        assert home["services"][name]["env_file"] == ".env.home"
        env = home["services"][name]["environment"]
        assert env["S3_ENDPOINT_URL"] == "http://minio:9000"
        assert env["S3_PUBLIC_ENDPOINT_URL"].startswith("${S3_PUBLIC_ENDPOINT_URL:?")


def test_minio_image_matches_dev_compose(home):
    dev = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    image = dev["services"]["minio"]["image"]
    assert "@sha256:" in image
    assert home["services"]["minio"]["image"] == image
    assert home["services"]["minio-init"]["image"] == image


def test_minio_cors_follows_cors_origins(home):
    env = home["services"]["minio"]["environment"]
    assert env["MINIO_API_CORS_ALLOW_ORIGIN"].startswith("${CORS_ORIGINS:?")


def test_passwords_are_required_without_defaults():
    text = COMPOSE.read_text()
    for var in ("POSTGRES_PASSWORD", "MINIO_ROOT_PASSWORD", "MINIO_ROOT_USER", "MINIO_APP_USER",
                "MINIO_APP_PASSWORD"):  # fmt: skip
        uses = re.findall(r"\$\{" + var + r"([^}]*)\}", text)
        assert uses, var
        assert all(u.startswith(":?") for u in uses), (var, uses)  # 기본값(:-) 없이 필수


def test_env_example_secrets_empty(env_example):
    for secret in ("POSTGRES_PASSWORD", "MINIO_ROOT_PASSWORD", "MINIO_APP_PASSWORD", "JWT_SECRET",
                   "OPENAI_API_KEY", "S3_PUBLIC_ENDPOINT_URL"):  # fmt: skip
        assert env_example[secret] == "", secret
    assert env_example["BIND_ADDR"] == "127.0.0.1"
    assert (env_example["API_PORT"], env_example["MINIO_PORT"]) == ("8000", "9000")
    assert env_example["CORS_ORIGINS"].split(",")[0] == "http://localhost:5173"


def test_env_home_is_gitignored_but_example_is_not():
    def ignored(name):
        res = subprocess.run(  # noqa: S603
            ["git", "check-ignore", "-q", name],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
        )
        return res.returncode == 0

    assert ignored(".env.home")
    assert not ignored(".env.home.example")


def test_home_update_script():
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)  # noqa: S603, S607
    order = [
        text.index("git pull --ff-only"),
        text.index("dc build"),
        text.index("alembic upgrade head"),
        text.index("dc up -d --remove-orphans"),
        text.index("== 상태 확인 =="),  # 이 뒤에서 /api/v1/health를 확인
    ]
    assert "/api/v1/health" in text[text.index("== 상태 확인 ==") :]
    assert order == sorted(order)
    assert 'exec bash "$SELF" --no-pull' in text  # pull 뒤 새 스크립트로
    assert "docker-compose.home.yml" in text and "docker-compose.prod.yml" not in text


def test_app_uses_dedicated_minio_account_not_root(home):
    for name in ("api", "worker"):
        env = home["services"][name]["environment"]
        assert env["S3_ACCESS_KEY"].startswith("${MINIO_APP_USER:?")
        assert env["S3_SECRET_KEY"].startswith("${MINIO_APP_PASSWORD:?")
        assert "MINIO_ROOT" not in str(env)
    init = home["services"]["minio-init"]
    script = init["command"][0]
    # 버킷 한정 정책을 만들어 앱 계정에 붙인다
    assert "arn:aws:s3:::$$S3_BUCKET/*" in script
    assert "mc admin user add" in script and "mc admin policy attach" in script
    # 관리자 비밀번호는 명령 인자가 아니라 환경 변수(MC_HOST_*)로
    assert "MINIO_ROOT_PASSWORD" not in script
    assert init["environment"]["MC_HOST_local"].startswith("http://${MINIO_ROOT_USER:?")


def test_bucket_default_matches_settings(home):
    from app.core.config import Settings

    default = Settings.model_fields["s3_bucket"].default
    assert (
        home["services"]["minio-init"]["environment"]["S3_BUCKET"] == f"${{S3_BUCKET:-{default}}}"
    )
    assert home["services"]["api"]["environment"]["S3_BUCKET"] == f"${{S3_BUCKET:-{default}}}"


def test_home_update_checks_worker_and_scopes_cleanup():
    text = SCRIPT.read_text()
    assert "inspect ping" in text  # 워커가 살아 있는지
    assert "curl -fsS" in text  # 호스트 포트로도 접속되는지
    # 같은 서버의 다른 서비스 이미지·빌드 캐시는 지우지 않는다
    prunes = [ln for ln in text.splitlines() if "prune" in ln and not ln.lstrip().startswith("#")]
    assert prunes == [
        "docker image prune -f --filter label=com.docker.compose.project=presento-home >/dev/null"
    ]
