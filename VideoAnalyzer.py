
class VideoAnalyzer(object):
    """
    Look for fixed-angle video sequences in a video
    """
    def __init__(self,
                 model      ='alexnet',
                 metric     = '',
                 stabilizer = None):

        self.model      = model
        self.metric     = metric
        self.stabilizer = stabilizer
        self.frame_rate = -1

    def process(self, video, sequences=50):
        # Get the frame rate
        self.frame_rate = 30

        # Calculate total frames
        self.total_frames = len(video)

        # Iterate in chunks of video
        #    Get first and last frame
        #    Calculate feature maps of the two frames
        #
