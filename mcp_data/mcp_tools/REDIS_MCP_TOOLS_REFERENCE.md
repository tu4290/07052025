# Redis MCP Tools Reference

## Overview
**Server Name**: `mcp.config.usrlocalmcp.redis`  
**Status**: ❌ NON-FUNCTIONAL (Connection Issues)  
**Purpose**: High-performance key-value storage and caching layer  
**Last Updated**: 2024-12-20  

## Server Description
The Redis MCP provides high-performance key-value storage capabilities for caching, session management, and temporary data storage. It leverages Redis's in-memory data structure store for optimal performance in data-intensive operations.

## Available Tools

### 1. set
**Description**: Set a Redis key-value pair with optional expiration

**Status**: ❌ Non-functional (Connection closed)

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "key": {
      "type": "string",
      "description": "Redis key"
    },
    "value": {
      "type": "string",
      "description": "Value to store"
    },
    "expireSeconds": {
      "type": "number",
      "description": "Optional expiration time in seconds"
    }
  },
  "required": ["key", "value"]
}
```

**Parameters**:
- `key` (string, required): The Redis key identifier
- `value` (string, required): The value to store
- `expireSeconds` (number, optional): Expiration time in seconds

**Usage Examples**:
```javascript
// Basic key-value storage
{
  "key": "user_session_123",
  "value": "active_session_data"
}

// With expiration (1 hour)
{
  "key": "temp_cache_data",
  "value": "cached_result",
  "expireSeconds": 3600
}

// Market data caching
{
  "key": "options_chain_AAPL",
  "value": "{\"strike\":150,\"delta\":0.5}",
  "expireSeconds": 300
}
```

**Expected Response**:
```json
{
  "success": true,
  "message": "Key set successfully"
}
```

**Current Error**:
```
MCP error -32000: Connection closed
```

## Use Cases for Elite Options System

### 1. Market Data Caching
- Cache real-time options prices
- Store calculated Greeks temporarily
- Cache market volatility data
- Store intraday price movements

### 2. Session Management
- User authentication tokens
- Dashboard state persistence
- User preferences and settings
- Active trading session data

### 3. Performance Optimization
- Cache expensive calculations
- Store frequently accessed data
- Reduce database query load
- Improve response times

### 4. Temporary Data Storage
- Store intermediate calculation results
- Cache API responses
- Store real-time alerts
- Temporary file processing data

## Integration Patterns

### Pattern 1: Market Data Caching
```javascript
// Cache options chain data
redis.set({
  key: `options_chain_${symbol}_${expiry}`,
  value: JSON.stringify(optionsData),
  expireSeconds: 300 // 5 minutes
});
```

### Pattern 2: Session Management
```javascript
// Store user session
redis.set({
  key: `session_${userId}`,
  value: JSON.stringify(sessionData),
  expireSeconds: 86400 // 24 hours
});
```

### Pattern 3: Calculation Caching
```javascript
// Cache expensive calculations
redis.set({
  key: `greeks_${symbol}_${strike}_${expiry}`,
  value: JSON.stringify(greeksData),
  expireSeconds: 60 // 1 minute
});
```

## Configuration Requirements

### Redis Server Setup
1. **Installation**: Redis server must be installed and running
2. **Configuration**: Proper connection parameters required
3. **Authentication**: Configure Redis authentication if needed
4. **Network**: Ensure Redis is accessible from MCP server
5. **Memory**: Allocate sufficient memory for caching needs

### MCP Configuration
1. **Connection String**: Configure Redis connection parameters
2. **Authentication**: Set up Redis credentials if required
3. **Timeout Settings**: Configure connection and operation timeouts
4. **Error Handling**: Implement proper error handling and fallbacks

## Troubleshooting

### Current Issue: Connection Closed
**Problem**: MCP error -32000: Connection closed

**Possible Causes**:
1. Redis server not running
2. Incorrect connection parameters
3. Network connectivity issues
4. Authentication failures
5. Firewall blocking connections

**Diagnostic Steps**:
1. Verify Redis server status
2. Test Redis connectivity with redis-cli
3. Check MCP configuration parameters
4. Verify network connectivity
5. Review Redis logs for errors

**Resolution Steps**:
1. Start Redis server if not running
2. Verify and correct connection parameters
3. Test Redis connectivity independently
4. Update MCP configuration if needed
5. Restart MCP server after configuration changes

## Performance Considerations

### Memory Management
- Monitor Redis memory usage
- Implement appropriate expiration policies
- Use memory-efficient data structures
- Regular cleanup of expired keys

### Connection Pooling
- Implement connection pooling for high-load scenarios
- Configure appropriate connection limits
- Monitor connection usage patterns
- Implement connection retry logic

### Data Serialization
- Use efficient JSON serialization
- Consider compression for large data sets
- Implement proper data validation
- Handle serialization errors gracefully

## Security Considerations

### Data Protection
- Never store sensitive data without encryption
- Implement proper access controls
- Use secure connection protocols
- Regular security audits

### Authentication
- Configure Redis authentication
- Use strong passwords
- Implement role-based access
- Monitor access patterns

## Monitoring and Maintenance

### Health Checks
- Regular connectivity tests
- Memory usage monitoring
- Performance metrics tracking
- Error rate monitoring

### Maintenance Tasks
- Regular Redis server updates
- Memory optimization
- Configuration reviews
- Backup and recovery procedures

## Future Enhancements

### Additional Tools (Potential)
- `get`: Retrieve values by key
- `delete`: Remove keys from Redis
- `exists`: Check if key exists
- `expire`: Set expiration on existing keys
- `list`: List keys matching patterns

### Advanced Features
- Redis Streams for real-time data
- Pub/Sub for event notifications
- Redis Modules for specialized operations
- Cluster support for high availability

## Conclusion

The Redis MCP is currently non-functional due to connection issues but represents a critical component for high-performance caching and session management in the Elite Options System. Resolving the Redis server connectivity issues is essential for optimal system performance and user experience.

**Next Steps**:
1. Diagnose and resolve Redis server connectivity
2. Test Redis MCP functionality after server restoration
3. Implement caching strategies for market data
4. Monitor performance improvements with Redis integration