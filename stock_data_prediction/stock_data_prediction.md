# Stock data prediction using LSTM/GRU

## Preprocessing the data

#### User configuration
- FORMAT: data format to be stored
    - csv and parquet is supported
    - parquet is prefered for performance reason\
    <span style="color:red"> pyarrow package is required to use parquet format </span>

- COMPRESSION: Choosing compression method used in parquet format
    - snappy, gzip, brotli is supported
    - snappy is choosen to be default compression method

- PRECISION: precision of pre-processed data
    - 16, 32, 64 is supported
    - Specifies directory name of the pre-processed data will be saved. ex) pre_32

#### What do you expect to see
- Processing single data
    1. Choose single stocks name and drop 'Name'\
    <span style="color:red"> If given stocks name is not supported, it will raise error </span>

    2. Check missing values of data\
    When there is missing value(NaN), it will show how many and where they are

    3. Plotting the data\
    You may see 'Volume' column does not corresponds to other columns\
    Correlation between each other items and auto-corelation clearly shows that 'Volume' may not improve any data quality to predict stock prices

    4. Drop 'Volume' and fill missing values\
    Since other columns 'Open', 'High', 'Low', 'Close' are highly correlated, we can fill missing values as the average of non-missing values at same date

    5. Save the pre-processed data\
    file name will be STOCKS.FORMAT.COMPRESSION. ex)IBM.parquet.snappy
<br/>

- Process over every stock data available
    Skip the process of plotting

## Stock Data Prediction

#### User configuration
- STOCKS: name of stocks to be predicted
    <span style="color:red"> If given stocks name is not supported, it will raise error </span>

- PRECISON: precision of prediction
    - 16, 32, 64 is supported
    - A preprocessing process of the same precision must precede

- PRE_PROCESS: Whether to use pre-processed data from data_preprocessing.ipynb
    - Default value of True

- RNN_TYPE: type of rnn
    - LSTM, GRU is supported

- IN_FEATURES/OUT_FEATURES: features to be input/output of machine
    - Choose from 'Open', 'Close', 'High', 'Low'. Order does not matters

- HIDDEN_SIZE: Size of hidden variables inside LSTM/GRU module
    - Defualt value of 50

- NUM_LAYERS: Number of stacked layers of LSTM/GRU module
    - Default value of 4

- DROPOUT: drop out rate of NN
    - Default value of 0.2

- PAST_DAYS: Number of days to be used to predict
    - Default value of 60 days

- SUCCESSIVE_DAYS: Number of days to be predicted
    - Defualt value of 1 day

- OPTIMIZER_NAME: Name of optimizer to be used
    - Default value of 'RMSprop'
    - Use the name from torch.optim

- LEARNING_RATE: learning rate of optimizer
    - Default value of 1e-3

- L2_REGULARIZATION: L2 regularization factor for optimizer
    - Defualt value of 0

- SCALER_NAME: name of scaler to normalize data
    - Defualt sacler of MinMaxScaler
    - Choose the name from sklearn.preprocessing

#### What do you expect to see
1. Generate dataset and neural network class corresponds to user input
    - Brief summary of dataset: Number of train/validation/test set and their shapes
    - Brief summary of neural network: Shape of NN and number of trainable parameters
    - If you have already trained model, load them at this time

2. Train the model
    - If verbose=True, print train/validation loss for every epochs
    - Plot the loss history

3. Test the model
    - averge_test: Take the mean value of successive days for every prediction of model
    - recurrent_test: Recurrently take the output of model as new input for model
    - If verbose=True, print test loss
    - Plot the real value and average/recurrent test results

4. Save the model
    - If the model is well-trained, save the model
