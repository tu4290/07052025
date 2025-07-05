# Context7 MCP Tools Reference

This document provides a reference for the available tools within the Context7 MCP (Model Context Protocol) for the Elite Options System v2.5 project.

The Context7 MCP is responsible for specialized context management, advanced analysis, contextual intelligence, and pattern recognition within the system.

## Available Tools

Here is a list of the tools available in the Context7 MCP, along with their descriptions and usage examples:

1. **`resolve-library-id`**
   * **Description:** Resolves a package/product name to a Context7-compatible library ID and provides metadata about the library.
   * **CRITICAL**: Use `libraryName` parameter (NOT `package_name`)
   * **Usage Example:**

     ```json
     {
       "server_name": "mcp.config.usrlocalmcp.context7",
       "tool_name": "resolve-library-id",
       "args": {
         "libraryName": "pydantic-ai"
       }
     }
     ```

2. **`get-library-docs`**
   * **Description:** Fetches up-to-date documentation for a library. You must call 'resolve-library-id' first to get the proper library identifier.
   * **CRITICAL**: Use `context7CompatibleLibraryID` parameter (NOT `library_id`)
   * **Usage Example:**

     ```json
     {
       "server_name": "mcp.config.usrlocalmcp.context7",
       "tool_name": "get-library-docs",
       "args": {
         "context7CompatibleLibraryID": "/pydantic/pydantic-ai"
       }
     }
     ```

## Tool Schemas and Parameters

### resolve-library-id

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "libraryName": {
      "type": "string",
      "description": "The name of the package or library to resolve"
    }
  },
  "required": ["libraryName"]
}
```

**Note**: The actual implementation uses `libraryName` as the required parameter, not `package_name` as shown in some documentation examples.

### get-library-docs

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "context7CompatibleLibraryID": {
      "type": "string",
      "description": "The Context7-compatible library identifier from resolve-library-id"
    }
  },
  "required": ["context7CompatibleLibraryID"]
}
```

**Note**: The actual implementation uses `context7CompatibleLibraryID` as the required parameter, not `library_id` as shown in some documentation examples. Additional parameters like `section`, `format`, and `include_examples` are not supported in the current implementation.

## Context Types and Categories

### Library Resolution Context
- **Package Ecosystems:** npm (JavaScript), pypi (Python), maven (Java), nuget (.NET), gem (Ruby)
- **Version Handling:** Semantic versioning, latest stable, beta releases
- **Metadata:** Dependencies, compatibility, security information

### Documentation Context
- **API Documentation:** Function signatures, parameters, return types
- **Usage Examples:** Code snippets, best practices, common patterns
- **Integration Guides:** Setup instructions, configuration options
- **Troubleshooting:** Common issues, error handling, debugging tips

## Workflow Examples

### Example 1: Library Research and Documentation Retrieval

```json
[
  {
    "tool_name": "resolve-library-id",
    "parameters": {
      "package_name": "fastapi",
      "ecosystem": "pypi"
    }
  },
  {
    "tool_name": "get-library-docs",
    "parameters": {
      "library_id": "fastapi-0.104.1",
      "section": "routing",
      "format": "markdown",
      "include_examples": true
    }
  }
]
```

### Example 2: Multi-Library Compatibility Analysis

```json
[
  {
    "tool_name": "resolve-library-id",
    "parameters": {
      "package_name": "react",
      "version": "18.x",
      "ecosystem": "npm"
    }
  },
  {
    "tool_name": "resolve-library-id",
    "parameters": {
      "package_name": "typescript",
      "ecosystem": "npm"
    }
  },
  {
    "tool_name": "get-library-docs",
    "parameters": {
      "library_id": "react-18.2.0",
      "section": "typescript-integration",
      "format": "markdown"
    }
  }
]
```

## Usage Guidelines and Best Practices

### Library Resolution Best Practices
- Always specify the ecosystem when dealing with packages that exist in multiple ecosystems
- Use semantic versioning for production environments
- Verify library compatibility before integration
- Check security advisories and maintenance status

### Documentation Retrieval Best Practices
- Start with general documentation before diving into specific sections
- Include examples when learning new APIs or troubleshooting
- Use markdown format for better readability and integration
- Cache frequently accessed documentation for performance

### Context Management Guidelines
- Maintain version consistency across related libraries
- Document library choices and rationale for future reference
- Regular updates to ensure current documentation and security patches
- Cross-reference with project requirements and constraints

## Integration with Other MCPs

The Context7 MCP integrates with other MCPs to provide enhanced contextual intelligence:

### With Persistent Knowledge Graph
- Store library metadata and documentation insights for long-term project intelligence
- Track library evolution and compatibility changes over time
- Maintain relationships between libraries and project components

### With Sequential Thinking
- Provide contextual information for architectural decision-making
- Support step-by-step analysis of library integration challenges
- Enhance problem-solving with relevant documentation and examples

### With TaskManager
- Support task planning with accurate library capability information
- Provide documentation context for development tasks
- Assist in dependency management and upgrade planning

### With Memory MCP
- Cache recent library lookups and documentation for session continuity
- Maintain context of current development focus and library usage
- Track temporary decisions and research findings

### With Search MCPs (Exa/Brave)
- Validate library information with external sources
- Cross-reference documentation with community resources
- Supplement official docs with tutorials and best practices

### With Puppeteer MCP
- Automate library testing and validation processes
- Capture screenshots of documentation for visual reference
- Monitor library websites for updates and announcements

## Advanced Context Analysis Features

### Pattern Recognition
- Identify common library usage patterns across projects
- Detect potential compatibility issues before they occur
- Recognize architectural patterns and suggest improvements

### Contextual Intelligence
- Provide context-aware library recommendations
- Analyze project requirements against library capabilities
- Suggest alternative libraries based on project constraints

### Advanced Analysis Capabilities
- Dependency graph analysis and optimization
- Security vulnerability assessment across library stack
- Performance impact analysis of library choices
- License compatibility verification

## Error Handling and Troubleshooting

### Common Error Scenarios
- **Library Not Found:** Package name misspelling or ecosystem mismatch
- **Version Conflicts:** Incompatible version specifications
- **Documentation Unavailable:** Library too new or deprecated
- **Rate Limiting:** Too many rapid requests to documentation sources

### Error Resolution Strategies
- Verify package names and ecosystems using multiple sources
- Use version ranges instead of exact versions when appropriate
- Fall back to community documentation when official docs are unavailable
- Implement request throttling and caching for high-volume usage

### Debugging Tools
- Library metadata inspection for troubleshooting
- Documentation source verification and validation
- Context history tracking for issue reproduction
- Integration testing with resolved libraries

## Performance Considerations

### Optimization Strategies
- Cache resolved library IDs to avoid repeated lookups
- Batch documentation requests when possible
- Use incremental updates for large documentation sets
- Implement intelligent prefetching based on usage patterns

### Resource Management
- Monitor API rate limits and usage quotas
- Optimize documentation parsing and storage
- Balance between fresh data and performance
- Implement graceful degradation for service unavailability

## Security and Privacy

### Data Protection
- No sensitive project information sent to external documentation services
- Library metadata cached locally when possible
- Secure handling of authentication tokens for private repositories
- Regular security audits of resolved libraries

### Privacy Considerations
- Minimal data sharing with external documentation providers
- Anonymous usage patterns when accessing public documentation
- Respect for library maintainer privacy and terms of service
- Compliance with data protection regulations

## File Location and Maintenance

This documentation file is located at `mcp_data/taskmanager/CONTEXT7_MCP_TOOLS_REFERENCE.md` within the Elite Options System v2.5 project structure.

### Maintenance Procedures
- Regular updates to reflect Context7 MCP capability changes
- Validation of example workflows and parameter schemas
- Integration testing with other MCP tools
- Documentation of new features and capabilities
- Performance monitoring and optimization recommendations

### Version Control
- Track changes to tool capabilities and parameters
- Maintain backward compatibility documentation
- Document breaking changes and migration paths
- Regular review and validation of documented procedures

## Elite Options System v2.5 Specific Use Cases

### Financial Library Integration
- Resolve and document financial data processing libraries
- Analyze compatibility with options trading algorithms
- Provide context for quantitative analysis library selection
- Support integration of market data processing frameworks

### Dashboard Development Context
- Library documentation for React/TypeScript dashboard components
- Context analysis for charting and visualization libraries
- Integration guidance for real-time data display frameworks
- Performance optimization context for high-frequency updates

### Analytics Engine Support
- Mathematical and statistical library documentation
- Machine learning framework integration context
- Performance analysis library selection and optimization
- Scientific computing library compatibility assessment

---

*This reference document is part of the Elite Options System v2.5 MCP infrastructure and should be updated as the Context7 MCP evolves and new capabilities are added.*