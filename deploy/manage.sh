#!/usr/bin/env bash
# Auto-Ingest Docker Management Script
# Usage: ./deploy/manage.sh {start|stop|restart|logs|status|build|enqueue|run|health} [args...]

set -euo pipefail
COMPOSE_FILE="$(cd "$(dirname "$0")/.." && pwd)/docker-compose.yml"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_DIR"

usage() {
    cat <<EOF
Auto-Ingest Docker Management

Usage: $0 <command> [options]

Commands:
  start [services...]    Start services (default: all)
  stop [services...]     Stop services (default: all)
  restart [services...]  Restart services (default: all)
  logs [service]         Show logs (default: all, use -f for follow)
  status                 Show container status
  build                  Build/rebuild the Docker image
  health                 Check all service health
  clean                  Remove stopped containers and unused images

Examples:
  $0 start                 # Start all services
  $0 start ingest-service  # Start only ingest
  $0 logs -f ingest-service
  $0 health                # Check all services

Note: 'enqueue' and 'run' commands are deprecated — Job API has been removed
due to security concerns (unauthenticated HTTP → RCE). Use deploy/create_job.sh
or docker compose directly instead.
EOF
    exit 1
}

cmd_start() {
    local services="${1:-}"
    if [[ -n "$services" ]]; then
        echo "Starting: $services"
        docker compose -f "$COMPOSE_FILE" up -d "$services"
    else
        echo "Starting all services..."
        docker compose -f "$COMPOSE_FILE" up -d
    fi
    echo ""
    echo "Services started. Run '$0 status' to check."
}

cmd_stop() {
    local services="${1:-}"
    if [[ -n "$services" ]]; then
        echo "Stopping: $services"
        docker compose -f "$COMPOSE_FILE" stop "$services"
    else
        echo "Stopping all services..."
        docker compose -f "$COMPOSE_FILE" stop
    fi
    echo "Services stopped."
}

cmd_restart() {
    local services="${1:-}"
    if [[ -n "$services" ]]; then
        echo "Restarting: $services"
        docker compose -f "$COMPOSE_FILE" restart "$services"
    else
        echo "Restarting all services..."
        docker compose -f "$COMPOSE_FILE" restart
    fi
    echo "Services restarted."
}

cmd_logs() {
    local follow=false
    local services=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--follow) follow=true; shift ;;
            *) services="$1"; shift ;;
        esac
    done

    if [[ "$follow" == true ]]; then
        if [[ -n "$services" ]]; then
            docker compose -f "$COMPOSE_FILE" logs -f "$services"
        else
            docker compose -f "$COMPOSE_FILE" logs -f
        fi
    else
        local tail="${TAIL:-100}"
        if [[ -n "$services" ]]; then
            docker compose -f "$COMPOSE_FILE" logs --tail="$tail" "$services"
        else
            docker compose -f "$COMPOSE_FILE" logs --tail="$tail"
        fi
    fi
}

cmd_status() {
    echo "=== Auto-Ingest Services ==="
    echo ""
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
}

cmd_build() {
    echo "Building Docker image..."
    docker compose -f "$COMPOSE_FILE" build --no-cache
    echo "Build complete."
}

cmd_enqueue() {
    local kind="${1:-all}"
    if [[ ! "$kind" =~ ^(audio|dashcam|bodycam|all)$ ]]; then
        echo "Error: kind must be audio|dashcam|bodycam|all" >&2
        exit 1
    fi
    # DEPRECATED: Job API removed (unauthenticated, RCE risk). Use deploy/create_job.sh directly.
    echo "enqueue is deprecated — Job API has been removed." >&2
    echo "Use deploy/create_job.sh <kind> instead." >&2
    exit 1
}

cmd_run() {
    local type="${1:-ingest}"
    if [[ ! "$type" =~ ^(ingest|sync)$ ]]; then
        echo "Error: type must be ingest|sync" >&2
        exit 1
    fi
    # DEPRECATED: Job API removed (unauthenticated, RCE risk). Run services manually or via docker compose.
    echo "run is deprecated — Job API has been removed." >&2
    echo "Run 'docker compose up -d <service>' directly instead." >&2
    exit 1
}

cmd_health() {
    echo "=== Health Check ==="
    echo ""

    # Check containers
    echo "--- Containers ---"
    docker compose -f "$COMPOSE_FILE" ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || echo "docker compose failed"
    echo ""

    # Check Neo4j
    echo "--- Neo4j ---"
    if docker inspect --format='{{.State.Health.Status}}' neo4j 2>/dev/null | grep -q healthy; then
        echo "Neo4j: HEALTHY"
    else
        echo "Neo4j: UNHEALTHY or not running"
    fi
    echo ""

    # Check NAS mount
    echo "--- NAS Mount ---"
    local root="${NAS_ROOT:-/media/scott/NAS3}"
    if mountpoint -q "$root" 2>/dev/null || [[ -d "$root" ]]; then
        echo "NAS root ($root): MOUNTED"
        df -h "$root" 2>/dev/null | tail -1
    else
        echo "NAS root ($root): NOT MOUNTED"
    fi
}

cmd_clean() {
    echo "Removing stopped containers..."
    docker compose -f "$COMPOSE_FILE" down
    echo "Cleaning up unused images..."
    docker image prune -f
    echo "Done."
}

# Main
if [[ $# -lt 1 ]]; then
    usage
fi

command="$1"
shift

case "$command" in
    start)   cmd_start "$@" ;;
    stop)    cmd_stop "$@" ;;
    restart) cmd_restart "$@" ;;
    logs)    cmd_logs "$@" ;;
    status)  cmd_status ;;
    build)   cmd_build ;;
    enqueue) cmd_enqueue "$@" ;;
    run)     cmd_run "$@" ;;
    health)  cmd_health ;;
    clean)   cmd_clean ;;
    help|--help|-h) usage ;;
    *)
        echo "Unknown command: $command" >&2
        usage
        ;;
esac
