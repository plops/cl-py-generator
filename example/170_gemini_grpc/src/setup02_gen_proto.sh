export GDIR=~/src/googleapis
uv run python -m grpc_tools.protoc -I $GDIR \
    --python_out=. --grpc_python_out=. \
    $GDIR/google/ai/generativelanguage/v1beta/*.proto
