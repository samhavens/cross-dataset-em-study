#!/bin/bash

# Create source.xml with all Python files from root and src/
echo "Creating source.xml with all Python files..."

# Start XML file
cat > source.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<source_code>
EOF

# Function to process Python files
process_python_file() {
    local file="$1"
    echo "  <file path=\"$file\">" >> source.xml
    echo "    <![CDATA[" >> source.xml
    cat "$file" >> source.xml
    echo "    ]]>" >> source.xml
    echo "  </file>" >> source.xml
}

# Find and process all Python files in root directory
echo "Processing Python files in root directory..."
find . -maxdepth 1 -name "*.py" -type f | sort | while read -r file; do
    echo "Adding $file"
    process_python_file "$file"
done

# Find and process all Python files in src/ directory
echo "Processing Python files in src/ directory..."
find src/ -name "*.py" -type f 2>/dev/null | sort | while read -r file; do
    echo "Adding $file"
    process_python_file "$file"
done

# Close XML file
echo "</source_code>" >> source.xml

echo "✅ source.xml created with all Python files"
echo "📁 File size: $(du -h source.xml | cut -f1)"
echo "📄 Total files: $(grep -c '<file path=' source.xml)"