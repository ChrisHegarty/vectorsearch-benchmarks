package com.searchscale.lucene.cuvs.benchmarks;


public class BenchmarkConfiguration {

  public String benchmarkID;
  public String datasetFile;
  public int indexOfVector;
  public String vectorColName;
  public int numDocs;
  public int vectorDimension;
  public String queryFile;
  public int numQueriesToRun;
  public int flushFreq;
  public int topK;
  public int numIndexThreads;
  public int cuvsWriterThreads;
  public int queryThreads;
  public boolean createIndexInMemory;
  public boolean cleanIndexDirectory;
  public boolean saveResultsOnDisk;
  public boolean hasColNames;
  public String algoToRun;
  public String groundTruthFile;
  public String cuvsIndexDirPath;
  public String hnswIndexDirPath;

  // HNSW parameters
  public int hnswMaxConn; // 16 default (max 512)
  public int hnswBeamWidth; // 100 default (max 3200)
  public int hnswVisitedLimit;

  // Cagra parameters
  public int cagraIntermediateGraphDegree; // 128 default
  public int cagraGraphDegree; // 64 default
  public int cagraITopK;
  public int cagraSearchWidth;



  public void debugPrintArguments() {
    System.out.println("Benchmark ID: " + benchmarkID);
    System.out.println("Dataset file used is: " + datasetFile);
    System.out.println("Index of vector field is: " + indexOfVector);
    System.out.println("Name of the vector field is: " + vectorColName);
    System.out.println("Number of documents to be indexed are: " + numDocs);
    System.out.println("Number of dimensions are: " + vectorDimension);
    System.out.println("Query file used is: " + queryFile);
    System.out.println("Number of queries to run: " + numQueriesToRun);
    System.out.println("Flush frequency (every n documents): " + flushFreq);
    System.out.println("TopK value is: " + topK);
    System.out.println("numIndexThreads is: " + numIndexThreads);
    //System.out.println("Lucene HNSW threads: " + hnswThreads);
//    System.out.println("cuVS Merge strategy: " + mergeStrategy);
    System.out.println("Query threads: " + queryThreads);
    System.out.println("Create index in memory: " + createIndexInMemory);
    System.out.println("Clean index directory: " + cleanIndexDirectory);
    System.out.println("Save results on disk: " + saveResultsOnDisk);
    System.out.println("Has column names in the dataset file: " + hasColNames);
    System.out.println("algoToRun {Choices: HNSW | CAGRA | ALL}: " + algoToRun);
    System.out.println("Ground Truth file used is: " + groundTruthFile);
    System.out.println("CuVS index directory path is: " + cuvsIndexDirPath);
    System.out.println("HNSW index directory path is: " + hnswIndexDirPath);
    
    System.out.println("------- algo parameters ------");
    System.out.println("hnswMaxConn: " + hnswMaxConn);
    System.out.println("hnswBeamWidth: " + hnswBeamWidth);
    System.out.println("hnswVisitedLimit: " + hnswVisitedLimit);
    System.out.println("cagraIntermediateGraphDegree: " + cagraIntermediateGraphDegree);
    System.out.println("cagraGraphDegree: " + cagraGraphDegree);
    System.out.println("cagraITopK: " + cagraITopK);
    System.out.println("cagraSearchWidth: " + cagraSearchWidth);
  }
}