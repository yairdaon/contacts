#!/bin/bash

# Job monitoring script for SLURM and remote server jobs
# Monitors running jobs and displays both output messages and errors

# Configuration
REMOTE_HOST="server"
REMOTE_DIR="contacts"
REFRESH_INTERVAL=30  # seconds between updates
LOG_LINES=20         # number of recent log lines to show

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_section() {
    echo -e "\n${CYAN}--- $1 ---${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to check if we can connect to the remote server
check_connection() {
    if ! ssh -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connection test'" >/dev/null 2>&1; then
        print_error "Cannot connect to $REMOTE_HOST"
        echo "Please check:"
        echo "  - SSH connection is working"
        echo "  - Server is accessible"
        echo "  - SSH keys are configured properly"
        exit 1
    fi
}

# Function to get job status
get_job_status() {
    ssh "$REMOTE_HOST" "squeue -u \$(whoami) --format='%.10i %.15j %.8T %.10M %.6D %.20S %.20e'" 2>/dev/null
}

# Function to get recently completed jobs
get_recent_jobs() {
    # Get jobs from last 24 hours
    ssh "$REMOTE_HOST" "sacct --starttime=\$(date -d '1 day ago' '+%Y-%m-%d') --format=JobID,JobName,State,ExitCode,Start,End --parsable2" 2>/dev/null
}

# Function to display job output files
show_job_outputs() {
    local job_pattern="tests_*.out tests_*.err"
    
    print_section "Recent Job Output Files"
    
    # List available output files
    local files=$(ssh "$REMOTE_HOST" "cd $REMOTE_DIR && ls -t $job_pattern 2>/dev/null | head -10")
    
    if [ -z "$files" ]; then
        echo "No job output files found"
        return
    fi
    
    echo "Available output files (most recent first):"
    echo "$files" | nl
    
    echo -e "\n${CYAN}Showing FULL content from each recent file (pytest output with error rates):${NC}"
    
    # Show full content from recent files
    for file in $files; do
        echo -e "\n${YELLOW}=== $file ===${NC}"
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && cat '$file' 2>/dev/null" || echo "Could not read $file"
        echo -e "\n${YELLOW}=== End of $file ===${NC}"
    done
}

# Function to show detailed job info
show_job_details() {
    local job_id="$1"
    if [ -n "$job_id" ]; then
        print_section "Job Details for $job_id"
        ssh "$REMOTE_HOST" "scontrol show job $job_id" 2>/dev/null
    fi
}

# Function to tail live job output
tail_job_output() {
    local job_id="$1"
    if [ -n "$job_id" ]; then
        local out_file="${REMOTE_DIR}/tests_${job_id}.out"
        local err_file="${REMOTE_DIR}/tests_${job_id}.err"
        
        print_section "Live Output for Job $job_id"
        echo "Press Ctrl+C to stop following output"
        echo -e "${CYAN}Following $out_file and $err_file...${NC}\n"
        
        # Follow both stdout and stderr
        ssh "$REMOTE_HOST" "tail -f '$out_file' '$err_file' 2>/dev/null" || {
            print_warning "Output files not yet available for job $job_id"
        }
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -w, --watch           Watch jobs continuously (refresh every $REFRESH_INTERVAL seconds)"
    echo "  -o, --output          Show recent job output files"
    echo "  -j, --job JOB_ID      Show details for specific job ID"
    echo "  -f, --follow JOB_ID   Follow live output for specific job ID"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    Show current job status once"
    echo "  $0 -w                 Watch jobs continuously"
    echo "  $0 -o                 Show recent job outputs"
    echo "  $0 -j 12345           Show details for job 12345"
    echo "  $0 -f 12345           Follow live output for job 12345"
}

# Main monitoring function
monitor_once() {
    clear
    print_header "Job Monitor - $(date)"
    
    # Check connection
    if ! check_connection; then
        return 1
    fi
    
    # Show current running jobs
    print_section "Current Jobs"
    local jobs=$(get_job_status)
    if [ -n "$jobs" ]; then
        echo "$jobs"
        
        # Count jobs by status
        local running=$(echo "$jobs" | grep -c "RUNNING" || echo "0")
        local pending=$(echo "$jobs" | grep -c "PENDING" || echo "0")
        local total=$(echo "$jobs" | wc -l)
        
        echo -e "\n${GREEN}Running: $running${NC} | ${YELLOW}Pending: $pending${NC} | ${BLUE}Total: $total${NC}"
    else
        echo "No active jobs found"
    fi
    
    # Show recent completed jobs
    print_section "Recent Completed Jobs (Last 24h)"
    local recent=$(get_recent_jobs)
    if [ -n "$recent" ]; then
        echo "$recent" | grep -v "JobID|" | head -10 | while IFS='|' read -r jobid jobname state exitcode start end; do
            if [ "$state" = "COMPLETED" ]; then
                print_success "$jobid $jobname - Completed (Exit: $exitcode)"
            elif [ "$state" = "FAILED" ]; then
                print_error "$jobid $jobname - Failed (Exit: $exitcode)"
            else
                echo "$jobid $jobname - $state (Exit: $exitcode)"
            fi
        done
    else
        echo "No recent completed jobs found"
    fi
    
    # Show system load
    print_section "Server Status"
    local load=$(ssh "$REMOTE_HOST" "uptime" 2>/dev/null)
    echo "Server load: $load"
    
    local disk=$(ssh "$REMOTE_HOST" "df -h $REMOTE_DIR" 2>/dev/null | tail -1)
    echo "Disk usage: $disk"
}

# Parse command line arguments
case "$1" in
    -w|--watch)
        while true; do
            monitor_once
            echo -e "\n${CYAN}Press Ctrl+C to stop monitoring. Refreshing in $REFRESH_INTERVAL seconds...${NC}"
            sleep $REFRESH_INTERVAL
        done
        ;;
    -o|--output)
        check_connection
        show_job_outputs
        ;;
    -j|--job)
        if [ -z "$2" ]; then
            echo "Error: Job ID required"
            show_usage
            exit 1
        fi
        check_connection
        show_job_details "$2"
        ;;
    -f|--follow)
        if [ -z "$2" ]; then
            echo "Error: Job ID required"
            show_usage
            exit 1
        fi
        check_connection
        tail_job_output "$2"
        ;;
    -h|--help)
        show_usage
        ;;
    "")
        monitor_once
        ;;
    *)
        echo "Error: Unknown option $1"
        show_usage
        exit 1
        ;;
esac
