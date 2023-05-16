The Faceloglog algorithm is a modified [hyperloglog](https://www.youtube.com/watch?v=2PlrMCiUN_s&ab_channel=VictorSanchesPortella) algorithm used to count faces.
The algorithm uses an external library to get a vector from a face, after normalization the vector is fed to the hyperloglog.

For face database [click here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

## Install Dependencies
1. Install python 3.10 and add python to path.
2. In the terminal write `pip install -r requirements.txt`.


## How to use it:
#### Option A) one computer
1. Run the [camera_counting.py](camera_counting.py) by inputting `python camera_counting.py -c` to the terminal or use `-h` to see more options.
2. Input the log 2 of the number of buckets you need. The bigger the number the more people you expect to see.
3. locate the camera where people are likely to look.

#### Option B) server and client
1. run the [Server.py](Server.py) script in the terminal.
2. Input the log 2 of the number of buckets you need. The bigger the number the more people you expect to see.
3. run the [Client.py](Client.py) by inputting `python Client.py [address of server computer]` or `python Client.py` if it is on the local computer.
4. locate the cameras of the clients where people are likely to look.

#### Option C) you
create your own! the script is fairly simple and works well. the many classes allow you to create your own objects and interfaces. there are many to implement and improve, so you should try it for yourself.