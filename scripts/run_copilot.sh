#!/usr/bin/env bash
# Chạy Copilot Layer (web UI + API bridge) — độc lập với src/api.py.
#
# Cách dùng:
#   ./scripts/run_copilot.sh start    # http://127.0.0.1:8001
#   ./scripts/run_copilot.sh stop
#   ./scripts/run_copilot.sh restart
#   ./scripts/run_copilot.sh log
#
# Yêu cầu: API hệ thống cũ chạy ở cổng 8000 (./run_api.sh start).
set -e

PORT="${COPILOT_PORT:-8001}"
LOG=./copilot.log
PYTHON="${PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || PYTHON=python

start() {
    cd "$(dirname "$0")/.."

    if curl -s -m 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
        echo "Copilot đang chạy sẵn tại http://127.0.0.1:${PORT}"
        return 0
    fi

    if ! "${PYTHON}" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
        echo "Thiếu phụ thuộc. Cài bằng:"
        echo "    ${PYTHON} -m pip install -r mcp_copilot/requirements.txt"
        exit 1
    fi

    if command -v setsid >/dev/null 2>&1; then
        setsid nohup "${PYTHON}" -m uvicorn mcp_copilot.webapp:app \
            --host "${COPILOT_HOST:-127.0.0.1}" --port "${PORT}" \
            > "${LOG}" 2>&1 < /dev/null &
    else
        nohup "${PYTHON}" -m uvicorn mcp_copilot.webapp:app \
            --host "${COPILOT_HOST:-127.0.0.1}" --port "${PORT}" \
            > "${LOG}" 2>&1 < /dev/null &
    fi

    for _ in $(seq 1 25); do
        sleep 1
        if curl -s -m 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
            echo "OK — Copilot chạy tại http://127.0.0.1:${PORT}"
            echo "Giao diện chat: http://127.0.0.1:${PORT}/"
            echo "Tài liệu API:   http://127.0.0.1:${PORT}/docs"
            echo
            echo "Nhắc: API hệ thống cũ phải chạy ở cổng 8000 (./run_api.sh start)."
            return 0
        fi
    done

    echo "Lỗi khởi động. Xem log: tail -60 ${LOG}"
    exit 1
}

stop() {
    pkill -f "uvicorn mcp_copilot.webapp:app" >/dev/null 2>&1 || true
    echo "Đã tắt Copilot."
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    log)     tail -60 "${LOG}" ;;
    *)       echo "Dùng: ./scripts/run_copilot.sh {start|stop|restart|log}" ;;
esac
