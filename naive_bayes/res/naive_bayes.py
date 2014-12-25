#############################################################
# NaiveBayse, the class implement the naive bayse algorithm.#
#############################################################

class NaiveBayse:
    
    def __init__(self, data_path):
        training_data, meta_data= self.parse(data_path)
        y0_count = 0
        y1_count = 0
        x_y0_count = [0 for i in range(meta_data["vector_length"])]
        x_y1_count = [0 for i in range(meta_data["vector_length"])]
        for (output,x) in training_data:
            if output==0:
                y0_count += 1
                for i in range(meta_data["vector_length"]):
                    x_y0_count[i] = x_y0_count[i] + x[i]
            else:
                y1_count += 1
                for i in range(meta_data["vector_length"]):
                    x_y1_count[i] = x_y1_count[i] + x[i]
                    
        #probability vector for each features be 1 given y = 0
        self.alpha_x_y0 = [(float(x)+1)/(y0_count+2) for x in x_y0_count]
        #probability vector for each features be 1 given y = 1
        self.alpha_x_y1 = [(float(x)+1)/(y1_count+2) for x in x_y1_count]
        #probability of output being 1
        self.alpha_y1 = (float(y1_count)+1)/(y1_count+y0_count+1)

    
    """
    assume the format is correct. parse the file and return processable data
    @data_path: the absolute path to the file
    @return: an array of tuple and meta data. tuple has two elems,first is type
             number and the second is an array of numbers.
    """
    def parse(self, data_path):
        meta_data = dict()
        f = open(data_path)
        line = f.readline()
        while line != "!!!\n":
            (key,val) = line.split(":")
            meta_data[key] = int(val)
            line = f.readline()
        
        result = []
        for i in range(meta_data["size"]):
            line = f.readline()
            output, vector = line.split(":")
            output = int(output)
            feature_vector = [int(x) for x in vector.split(",")]
            if len(feature_vector) == meta_data["vector_length"]:
                result.append((output, feature_vector))
        f.close()
        return result, meta_data
        
    
    """
    predict the result given an input
    @x: an input feature vector(int).
    @return: return 1 or 0
    """
    def predict(self, x):
        #compute the probability of output being 1, using bayse rule(denominator is not calculated)
        prob_y1 = 1
        for i in range(len(x)):
            if x[i] == 1:
                prob_y1 *= self.alpha_x_y1[i] 
            else:
                prob_y1 *= 1-self.alpha_x_y1[i]
        prob_y1 *= self.alpha_y1
        
        #compute the probability of output being 1, using bayse rule(denominator is not calculated)
        prob_y0 = 1
        for i in range(len(x)):
            if x[i] == 1:
                prob_y0 *= self.alpha_x_y0[i] 
            else:
                prob_y0 *= 1-self.alpha_x_y0[i]
        prob_y0 *= 1-self.alpha_y1

        if prob_y0 > prob_y1:
            return 0
        else:
            return 1
