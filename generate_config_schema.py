#!/usr/bin/env python3
"""
Generate JSON Schema from EOTSConfigV2_5 Pydantic Model

This script generates a JSON schema file from the current EOTSConfigV2_5 Pydantic model,
ensuring the schema matches the actual model structure.
"""

import json
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from data_models.configuration_models import EOTSConfigV2_5
    
    def generate_schema():
        """Generate JSON schema from EOTSConfigV2_5 model."""
        try:
            # Generate schema using Pydantic v2 method
            schema = EOTSConfigV2_5.model_json_schema()
            
            # Add JSON Schema metadata
            schema["$schema"] = "http://json-schema.org/draft-07/schema#"
            schema["title"] = "EOTS v2.5 Configuration Schema"
            schema["description"] = "Comprehensive schema for validating the EOTS v2.5 application configuration file, generated from Pydantic models."
            
            return schema
            
        except Exception as e:
            print(f"Error generating schema: {e}")
            return None
    
    def save_schema(schema, output_path):
        """Save schema to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
            print(f"✅ Schema successfully generated and saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving schema: {e}")
            return False
    
    def main():
        """Main function to generate and save the schema."""
        print("🔄 Generating JSON schema from EOTSConfigV2_5 Pydantic model...")
        
        # Generate schema
        schema = generate_schema()
        if not schema:
            print("❌ Failed to generate schema")
            sys.exit(1)
        
        # Save to config directory
        output_path = project_root / "config" / "config.schema.v2_5.json"
        
        # Backup existing schema if it exists
        if output_path.exists():
            backup_path = output_path.with_suffix('.json.backup')
            output_path.rename(backup_path)
            print(f"📦 Existing schema backed up to: {backup_path}")
        
        # Save new schema
        if save_schema(schema, output_path):
            print("✅ Schema generation completed successfully!")
            
            # Print some basic info about the schema
            print(f"\n📊 Schema Statistics:")
            print(f"   - Total properties: {len(schema.get('properties', {}))}")
            print(f"   - Schema version: {schema.get('$schema', 'Unknown')}")
            print(f"   - Title: {schema.get('title', 'Unknown')}")
            
        else:
            print("❌ Failed to save schema")
            sys.exit(1)
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed and data_models is available")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
