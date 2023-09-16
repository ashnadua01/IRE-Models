## Assignment - 1
### Submitted By: Ashna Dua, 2021101072

#### Assumptions
1. The query, p, metric to be used for gathering top stems, is given from the command line using -q, -p and --metric respectively. Additionally --type is also used to mention about Boolean or Vector Model. 
2. In the vector model, the matrix stores the tf-idf as tf of the query term in the query times the idf in documents. 
3. In the boolean model, only simple queries involving "and", "or", and "not" are implemented. 
4. The Boolean representation of Query is of the type: ["Boolean Vector of term1", "Boolean Operator", "Boolean Vector of term2"] and so on.
5. Stems for both tf and tf-idf are implemented. The metric to be used can be given as input from the command line.