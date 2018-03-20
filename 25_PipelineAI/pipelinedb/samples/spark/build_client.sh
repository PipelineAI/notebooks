../../modeldb/scripts/gen_thrift_file.sh scala '../../modeldb/thrift/ModelDB.thrift' './src/main/thrift/' 
sbt clean && sbt assembly
