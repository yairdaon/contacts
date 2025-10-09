#!/bin/bash

# Script to kill all SLURM jobs and clean up output/error files

# Configuration
REMOTE_HOST="server"
REMOTE_DIR="contacts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check connection
check_connection() {
    if ! ssh -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connection test'" >/dev/null 2>&1; then
        print_error "Cannot connect to $REMOTE_HOST"
        exit 1
    fi
}

# Function to get current jobs
get_jobs() {
    ssh "$REMOTE_HOST" "squeue -u \$(whoami) --format='%.10i %.15j %.8T'" 2>/dev/null
}

# Function to kill all jobs
kill_all_jobs() {
    print_header "Killing All SLURM Jobs"
    
    # Get job IDs
    local job_ids=$(ssh "$REMOTE_HOST" "squeue -u \$(whoami) --format='%.10i' --noheader" 2>/dev/null | tr -d ' ')
    
    if [ -z "$job_ids" ]; then
        echo "No active jobs found to kill"
        return 0
    fi
    
    echo "Found jobs to kill:"
    echo "$job_ids"
    echo
    
    # Kill each job
    for job_id in $job_ids; do
        echo "Killing job $job_id..."
        ssh "$REMOTE_HOST" "scancel $job_id" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_success "Killed job $job_id"
        else
            print_error "Failed to kill job $job_id"
        fi
    done
    
    # Wait a moment for jobs to be cancelled
    sleep 2
    
    # Verify no jobs remain
    local remaining=$(ssh "$REMOTE_HOST" "squeue -u \$(whoami) --noheader" 2>/dev/null | wc -l)
    if [ "$remaining" -eq 0 ]; then
        print_success "All jobs killed successfully"
    else
        print_warning "$remaining jobs may still be running"
    fi
}

# Function to display output files before cleanup
show_output_files() {
    print_header "Displaying Job Output and Error Messages"
    
    # List files to be displayed
    local files=$(ssh "$REMOTE_HOST" "cd $REMOTE_DIR && ls -t tests_*.out tests_*.err 2>/dev/null")
    
    if [ -z "$files" ]; then
        echo "No output files found to display"
        return 0
    fi
    
    echo "Found output files (most recent first):"
    echo "$files" | nl
    echo
    
    # Display content from each file
    for file in $files; do
        echo -e "\n${YELLOW}=== $file ===${NC}"
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && cat '$file' 2>/dev/null" || echo "Could not read $file"
        echo -e "${YELLOW}=== End of $file ===${NC}\n"
    done
}

# Function to clean up output files
cleanup_files() {
    print_header "Cleaning Up Output Files"
    
    # List files to be deleted
    local files=$(ssh "$REMOTE_HOST" "cd $REMOTE_DIR && ls tests_*.out tests_*.err 2>/dev/null")
    
    if [ -z "$files" ]; then
        echo "No output files found to clean"
        return 0
    fi
    
    echo "Deleting output files..."
    
    # Count files
    local count=$(echo "$files" | wc -l)
    
    # Delete files
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && rm -f tests_*.out tests_*.err" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Deleted $count output files"
    else
        print_error "Failed to delete some output files"
    fi
    
    # Verify cleanup
    local remaining=$(ssh "$REMOTE_HOST" "cd $REMOTE_DIR && ls tests_*.out tests_*.err 2>/dev/null | wc -l")
    if [ "$remaining" -eq 0 ]; then
        print_success "All output files cleaned successfully"
    else
        print_warning "$remaining output files may still exist"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -j, --jobs-only       Kill jobs only (don't delete output files)"
    echo "  -f, --files-only      Delete output files only (don't kill jobs)" 
    echo "  -y, --yes             Skip confirmation prompt"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Default behavior kills all jobs AND deletes output files"
}

# Main execution
main() {
    local kill_jobs=true
    local cleanup=true
    local skip_confirm=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -j|--jobs-only)
                cleanup=false
                shift
                ;;
            -f|--files-only)
                kill_jobs=false
                shift
                ;;
            -y|--yes)
                skip_confirm=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check connection
    check_connection
    
    # Show current status
    echo "Current SLURM jobs:"
    get_jobs
    echo
    
    local files=$(ssh "$REMOTE_HOST" "cd $REMOTE_DIR && ls tests_*.out tests_*.err 2>/dev/null | wc -l")
    echo "Output files found: $files"
    echo
    
    # Confirmation prompt
    if [ "$skip_confirm" = false ]; then
        local action_desc=""
        if [ "$kill_jobs" = true ] && [ "$cleanup" = true ]; then
            action_desc="kill all jobs AND delete all output files"
        elif [ "$kill_jobs" = true ]; then
            action_desc="kill all jobs"
        elif [ "$cleanup" = true ]; then
            action_desc="delete all output files"
        fi
        
        echo -e "${YELLOW}Are you sure you want to $action_desc? (y/N)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Operation cancelled"
            exit 0
        fi
    fi
    
    # Execute requested actions in order: kill, show output, cleanup
    if [ "$kill_jobs" = true ]; then
        kill_all_jobs
        echo
    fi
    
    if [ "$cleanup" = true ]; then
        show_output_files
        echo
        cleanup_files
        echo
    fi
    
    print_success "Killer script completed successfully"
}

# Run main function with all arguments
main "$@"