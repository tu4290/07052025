# TaskManager MCP Data Directory

## Overview
This directory contains the data files and configuration for the TaskManager MCP (Model Context Protocol) server used in the Elite Options System v2.5.

## Directory Structure
```
mcp_data/taskmanager/
├── tasks.json          # Main task storage file
├── config.json         # TaskManager configuration
├── README.md          # This documentation
├── backups/           # Automatic backups (created as needed)
└── logs/              # Log files (created as needed)
```

## Safety Guardrails

### 1. Sandboxed Environment
- All TaskManager operations are confined to this project directory
- No access to system files outside the project scope
- All file operations are logged and auditable

### 2. Task Validation
- Maximum task depth of 5 levels to prevent infinite recursion
- Task dependency validation to prevent circular dependencies
- Automatic timeout for long-running tasks (60 minutes default)

### 3. Approval Workflows
- Critical tasks require explicit approval before execution
- Restricted operations are blocked by default
- All task modifications are logged with timestamps

### 4. Data Protection
- Automatic backups of task data every 30 seconds
- Maximum of 100 concurrent requests to prevent resource exhaustion
- Log rotation to prevent disk space issues

## Configuration

The `config.json` file contains all TaskManager settings including:
- Safety guardrails and restrictions
- Logging configuration
- Performance limits
- Allowed/restricted operations

## Usage

The TaskManager MCP tools can be used through the MCP interface:
- `list_requests`: View current tasks and their status
- `request_planning`: Create new task planning requests
- Additional tools as configured in the MCP server

## Security Notes

1. **File Access**: Limited to project directory only
2. **Network Access**: Requires approval for external connections
3. **System Operations**: Blocked for system-level changes
4. **Resource Limits**: Configured to prevent resource exhaustion

## Maintenance

- Log files are automatically rotated when they exceed 10MB
- Completed tasks are automatically cleaned up after 30 days
- Backup files are maintained for disaster recovery

## Troubleshooting

If the TaskManager MCP encounters issues:
1. Check the log files in the logs/ directory
2. Verify the tasks.json file is not corrupted
3. Ensure proper permissions on the mcp_data directory
4. Review the config.json for any misconfigurations

For additional support, refer to the Elite Options System v2.5 documentation.