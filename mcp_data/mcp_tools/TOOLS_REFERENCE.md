# TaskManager MCP Tools Reference

This document provides a comprehensive reference for all available tools in the TaskManager MCP server for the Elite Options System v2.5.

## Available Tools

### 1. request_planning
**Description:** Register a new user request and plan its associated tasks. You must provide both the original request and a list of tasks.

**Usage:**
```json
{
  "originalRequest": "Description of the main request",
  "tasks": [
    {
      "title": "Task title",
      "description": "Detailed task description",
      "priority": "high|medium|low"
    }
  ]
}
```

### 2. get_next_task
**Description:** Given a 'requestId', return the next pending task (not done yet). If all tasks are done, returns appropriate message.

**Usage:**
```json
{
  "requestId": "req_001"
}
```

### 3. mark_task_done
**Description:** Mark a given task as done after you've completed it. Provide 'requestId' and 'taskId'.

**Usage:**
```json
{
  "requestId": "req_001",
  "taskId": "task_001"
}
```

### 4. approve_task_completion
**Description:** Once the assistant has marked a task as done using 'mark_task_done', this tool allows approval of the completion.

**Usage:**
```json
{
  "requestId": "req_001",
  "taskId": "task_001"
}
```

### 5. approve_request_completion
**Description:** After all tasks are done and approved, this tool finalizes the entire request.

**Usage:**
```json
{
  "requestId": "req_001"
}
```

### 6. open_task_details
**Description:** Get details of a specific task by 'taskId'. This is for inspecting task information.

**Usage:**
```json
{
  "taskId": "task_001"
}
```

### 7. list_requests
**Description:** List all requests with their basic information and summary of tasks. This provides an overview of the system state.

**Usage:**
```json
{}
```

### 8. add_tasks_to_request
**Description:** Add new tasks to an existing request. This allows extending a request with additional work.

**Usage:**
```json
{
  "requestId": "req_001",
  "tasks": [
    {
      "title": "New task title",
      "description": "New task description",
      "priority": "high|medium|low"
    }
  ]
}
```

### 9. update_task
**Description:** Update an existing task's title and/or description. Only uncompleted tasks can be updated.

**Usage:**
```json
{
  "taskId": "task_001",
  "title": "Updated task title",
  "description": "Updated task description"
}
```

### 10. delete_task
**Description:** Delete a specific task from a request. Only uncompleted tasks can be deleted.

**Usage:**
```json
{
  "taskId": "task_001"
}
```

## Workflow Examples

### Basic Workflow
1. **Create Request:** Use `request_planning` to create a new request with tasks
2. **Get Next Task:** Use `get_next_task` to retrieve the next pending task
3. **Complete Task:** Perform the actual work
4. **Mark Done:** Use `mark_task_done` to mark the task as completed
5. **Approve Task:** Use `approve_task_completion` to approve the completion
6. **Repeat:** Continue with steps 2-5 for remaining tasks
7. **Finalize:** Use `approve_request_completion` to finalize the entire request

### Management Workflow
- **Monitor Progress:** Use `list_requests` to see all requests and their status
- **Inspect Details:** Use `open_task_details` to examine specific tasks
- **Extend Work:** Use `add_tasks_to_request` to add more tasks
- **Modify Tasks:** Use `update_task` to change task details
- **Remove Tasks:** Use `delete_task` to remove unnecessary tasks

## Task Priority Levels
- **high:** Critical tasks that should be completed first
- **medium:** Important tasks with moderate urgency
- **low:** Tasks that can be completed when time permits

## Task Status Values
- **pending:** Task is waiting to be started
- **in_progress:** Task is currently being worked on
- **done:** Task has been completed
- **approved:** Task completion has been approved

## Best Practices

1. **Clear Task Descriptions:** Always provide detailed descriptions for tasks
2. **Appropriate Priorities:** Use priority levels to guide task execution order
3. **Regular Monitoring:** Use `list_requests` to track overall progress
4. **Proper Approval Flow:** Follow the mark_done → approve_completion workflow
5. **Task Dependencies:** Consider task dependencies when planning execution order

## Integration with Elite Options System v2.5

The TaskManager MCP is designed to work seamlessly with the Elite Options System v2.5 architecture:

- **Analytics Tasks:** Breaking down complex analytics into manageable steps
- **Dashboard Development:** Managing UI/UX improvements and feature additions
- **Data Pipeline Work:** Organizing data processing and validation tasks
- **Testing Workflows:** Structuring comprehensive testing procedures
- **Deployment Tasks:** Managing system deployment and configuration

## Error Handling

If tools return errors:
- Check that required parameters are provided
- Verify that task/request IDs exist
- Ensure tasks are in the correct state for the operation
- Review the tasks.json file for data integrity

## File Locations

- **Tasks Data:** `mcp_data/taskmanager/tasks.json`
- **Configuration:** `mcp_data/taskmanager/config.json`
- **Logs:** `mcp_data/taskmanager/logs/`
- **Backups:** `mcp_data/taskmanager/backups/`

This reference document should be consulted whenever working with the TaskManager MCP to ensure proper tool usage and workflow management.