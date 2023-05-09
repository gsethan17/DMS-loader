from config import data_dir, meta_path, driver_states, total_drivers, cols_BIO, cols_CAN_conti, units
import pandas as pd
import numpy as np
import os
from glob import glob


        
class DataAnalyzer(object):
    '''A data analyzer for Driver Monitoring System Initialize the basic data to be loaded.
    Args: 
        drivers: `list` or `tuple`, the registered name of meta data. 
        Loads data only for passed driver(s).

        states: `list` or `tuple`, the predefined driver's states.
        Loads data only for passed state(s).
        Reference: "Multimodal data collection system for driver emotion recognition based on self-reporting in real-world driving." Oh, Geesung, et al., 2022, Sensors.

        name: `str`, the name of the data analyzer.

    ```python
    from utils import DataAnalyzer

    data_analyzer = DataAnalyzer()
    ```
    '''

    def __init__(self, drivers=[], states=[], name=None, **keyargs):

        if not type(drivers)==list: raise TypeError("drivers must be list.")
        if not type(states)==list: raise TypeError("states must be list.")

        self.drivers = list(drivers)
        self.states = list(states)
        if len(self.drivers) == 0: self.drivers = list(total_drivers)
        
        driver_states_reverse = dict(zip(driver_states.values(), driver_states.keys()))
        self.states = list(driver_states.keys()) if len(self.states) == 0 else [driver_states_reverse.get(x) for x in self.states]

        self.name = name

        self.meta_data = pd.read_csv(meta_path)
        self.meta_data = self.meta_data[self.meta_data['driver'].isin(self.drivers)].reset_index(inplace=False, drop=True)

        # load HIM info
        self.HMI_data = self.load_HMI_data()

    def load_HMI_data(self):
        """load data that self-reported by the driver during driving
        Args:
            None

        return:
            `dataframe`, shape=(num_shmple, num_col),
            columns are `time`, `driver`, `the predefined driver's states in integer type`
        """
        HMI_data = pd.DataFrame()

        for i in range(len(self.meta_data)):
            if self.meta_data.loc[i, 'HMI']:
                odo = str(self.meta_data.loc[i, 'start_odo'])
                driver = self.meta_data.loc[i, 'driver']

                path_HMIs = glob(os.path.join(data_dir, odo, 'HMI', '*.csv'))
                if not len(path_HMIs) == 1: raise FileExistsError(os.path.join(data_dir, odo, 'HMI'))
                df = pd.read_csv(path_HMIs[0])
                df['driver'] = driver

                HMI_data = pd.concat([HMI_data, df], axis=0)
        HMI_data.columns = ['time', 'driver', 'state']
        HMI_data = HMI_data[HMI_data['state'].isin(self.states)].reset_index(inplace=False, drop=True)

        return HMI_data.reset_index(inplace=False, drop=True)



    def plot_HMI_count(self, byDriver=True):
        """plot the number of self-reported driver's states as a bat chart
        Args:
            byDriver: `bool`, default:True
                If True, plot the bat chart for each driver.

        retern:
            `DataFrame`, the count of self-reporting by state(s) and driver(s).
        """
        condi_state = (lambda state: self.HMI_data['state']==state)
        condi_driver = (lambda driver: self.HMI_data['driver']==driver)

        show_dict = {
            'state':[],
            'driver':[],
            'count':[],
        }

        for state in self.states:
            for driver in self.drivers:
                count = len(self.HMI_data[condi_state(state) & condi_driver(driver)])

                show_dict['state'].append(state)
                show_dict['driver'].append(driver)
                show_dict['count'].append(count)

        import seaborn as sns
        import matplotlib.pyplot as plt
        if byDriver:
            sns.barplot(data=show_dict, x='state', y='count', hue='driver')
        else:
            sns.barplot(data=show_dict, x='state', y='count')
        plt.xticks([x for x in range(len(self.states))], [driver_states.get(x) for x in sorted(self.states)], fontsize=7)
        plt.show()

        df = pd.DataFrame(show_dict)
        df['state'] = df['state'].apply(lambda x : driver_states.get(x))

        return df

    def load_can_data(self, col):
        df_concat = pd.DataFrame()
            
        odos = list(self.meta_data[self.meta_data['CAN'] == True]['start_odo'])
        for odo in odos:
            file_path = glob(os.path.join(data_dir, str(odo), 'CAN', '*.csv'))
            if not len(file_path) == 1: raise FileExistsError(os.path.join(data_dir, str(odo), 'CAN'))
            df = pd.read_csv(file_path[0], low_memory=False)
            df = df[['timestamp', col]]
            if col == 'SAS_Angle' or col == 'LAT_ACCEL' or col == 'YAW_RATE':
                df[col] = df[col].abs()
            df = df.dropna(axis=0)
            df_concat = pd.concat([df_concat, df], axis=0)

        return df_concat.reset_index(inplace=False, drop=True)


    def load_bio_data(self, col):
        df_concat = pd.DataFrame()
        
        odos = list(self.meta_data[self.meta_data['bio'] == True]['start_odo'])
        for odo in odos:
            file_path = os.path.join(data_dir, str(odo), 'bio', col+'.csv')
            if not os.path.isfile(file_path): raise FileExistsError(os.path.join(data_dir, str(odo), 'bio'))

            df = pd.read_csv(file_path, header=None)

            str_time = df.loc[0].mean()
            hz = df.loc[1].mean()

            col_mean = df.loc[2:].mean(axis=1)
            if col == 'ACC':
                col_mean = np.sqrt((df.loc[2:]/64. * df.loc[2:]/64.).sum(axis=1))
                
            timestamp = [str_time + (x * (1/hz)) for x in range(len(col_mean))]

            df_ = pd.DataFrame({'timestamp':timestamp, col:col_mean})

            df_concat = pd.concat([df_concat, df_], axis=0)

        return df_concat.reset_index(inplace=False, drop=True)


    def check_time(self, utc_time):
        from datetime import datetime, timedelta
        kst_str_time = datetime.utcfromtimestamp(utc_time) + timedelta(hours=9)
        print(kst_str_time)

    def load_video_info(self):
        df_concat = pd.DataFrame()
            
        odos = list(self.meta_data[self.meta_data['video'] == True]['start_odo'])
        for odo in odos:
            file_path = glob(os.path.join(data_dir, str(odo), 'video', '*.csv'))
            if not len(file_path) == 1: raise FileExistsError(os.path.join(data_dir, str(odo), 'CAN'))
            df = pd.read_csv(file_path[0])
            df['dir_path'] = os.path.dirname(file_path[0])
            df['frame'] = df.index
            df_concat = pd.concat([df_concat, df], axis=0)

        return df_concat.reset_index(inplace=False, drop=True)
    
    def plot_mean_value(self, col, prev_sec, byDriver=True):
        """plot the mean of mean value of a given column for a period time until each state is reported as a bar chart.
        
        Args:
            col: `str`, data feature.
            prev_sec: `integer` or `float`, time range to observe until state is self-reported.
            byDriver: `bool`, default:True
                If True, plot the bat chart for each driver.

        retern:
            `DataFrame`, the mean value of a given column by state(s) and driver(s).
        """
        col = str(col)

        if col in cols_BIO:
            if col == 'IBI': raise ValueError("{} is not supported yet.".format(col))
            df = self.load_bio_data(col)
        
        elif col in cols_CAN_conti:
            df = self.load_can_data(col)
        
        else:
            raise ValueError("Invalid argument:", col)
        
        show_dict = {
            'state':[],
            'driver':[],
            'count':[],
            'mean_value':[],
        }
        
        condi_state = (lambda state: self.HMI_data['state']==state)
        condi_driver = (lambda driver: self.HMI_data['driver']==driver)
        
        condi_time = (lambda timestamp: df['timestamp']<=timestamp)
        condi_time_prev = (lambda timestamp: df['timestamp']>=timestamp-prev_sec)
        
        for state in self.states:
            for driver in self.drivers:
                time_list = self.HMI_data[condi_state(state) & condi_driver(driver)]['time']
                for timestamp in time_list:
                    mean_value = df[condi_time(timestamp) & condi_time_prev(timestamp)][col].mean()
                    show_dict['state'].append(state)
                    show_dict['driver'].append(driver)
                    show_dict['count'].append(len(time_list))
                    show_dict['mean_value'].append(mean_value)
                    
                
        import seaborn as sns
        import matplotlib.pyplot as plt
        if byDriver:
            sns.barplot(data=show_dict, x='state', y='mean_value', hue='driver')
        else:
            sns.barplot(data=show_dict, x='state', y='mean_value')
        plt.ylabel("{} [{}]".format(col, units[col]))
        plt.xticks([x for x in range(len(self.states))], [driver_states.get(x) for x in sorted(self.states)], fontsize=7)
        plt.show()

        df = pd.DataFrame(show_dict)
        df['state'] = df['state'].apply(lambda x : driver_states.get(x))
        
        return df
    
    def get_image(self, path, frames, view, mode):
        if not view in ['front', 'side']: raise ValueError("view argument should be among the 'front' or 'side'")
        if not mode in ['color', 'ir']: raise ValueError("view argument should be among the 'color' or 'ir'")
        
        import cv2
        
        video_path = glob(os.path.join(path, '*', '{}_{}2.avi'.format(mode, view)))[0]
        video = cv2.VideoCapture(video_path)
        
        imgs = []
        
        for frame in frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            
            flag, img = video.read()
            
            if not flag : raise ValueError("{} video has no {} frame".format(path, frame))
            
            if mode == 'color':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            imgs.append(img)
            
        return imgs
            
    
    def show_mean_image(self, view, mode, prev_sec, byDriver=True, showEach=False):
        """show the mean of mean image of a given view and mode camera for a period time until each state is reported.
        
        Args:
            view: `str`, 'front' or 'side', position of camera installation.
            mode: `str`, 'color' or 'ir', RGB camera of IR camera.
            prev_sec: `integer` or `float`, time range to observe until state is self-reported.
            byDriver: `bool`, default:True
                If True, show the image for each driver.
            showEach: `bool`, default:False
                If True, show the image for each sample.

        retern:
            mean_img: `dictionary`, mean of mean image by state(s) and driver(s).
            image_info: `dictionary`, all the video path and frame number by state(s) and driver(s).
        """
        df = self.load_video_info()
        
        image_info = {
            'state':[],
            'driver':[],
            'path':[],
            'frames':[],
        }
        
        mean_img = {}
        num_mean_imgs = {}
        for state in self.states:
            mean_img[state] = {}
            num_mean_imgs[state] = {}
            for driver in self.drivers:
                mean_img[state][driver] = np.zeros((720, 1280, 3))
                num_mean_imgs[state][driver] = 0
        
        condi_state = (lambda state: self.HMI_data['state']==state)
        condi_driver = (lambda driver: self.HMI_data['driver']==driver)
        
        condi_time = (lambda timestamp: df['timestamp']<=timestamp)
        condi_time_prev = (lambda timestamp: df['timestamp']>=timestamp-prev_sec)
        
        for state in self.states:
            for driver in self.drivers:
                time_list = self.HMI_data[condi_state(state) & condi_driver(driver)]['time']
                for timestamp in time_list:
                    paths = df[condi_time(timestamp) & condi_time_prev(timestamp)]['dir_path']
                    frames = df[condi_time(timestamp) & condi_time_prev(timestamp)]['frame']
                    if len(paths) == 0:
                        continue
                        # self.check_time(timestamp)
                        # raise FileExistsError("There is no matched image. {}, {}".format(driver, state))
                    if len(set(paths)) > 1:
                        raise FileExistsError("Among the extracted images, addresses that are not identical exist. \n{}".format(set(paths)))
                    
                        
                    path = list(set(paths))[0]
                    imgs = self.get_image(path, frames, view, mode)
                    num_img = len(np.array(imgs))
                    num_mean_imgs[state][driver] += num_img
                    each_mean_img = np.array(imgs).mean(axis=0)
                    
                    if mean_img[state][driver].sum() == 0:
                        mean_img[state][driver] = each_mean_img
                        
                    else:
                        mean_img[state][driver] = ((mean_img[state][driver] + each_mean_img) / 2.)
                        
                    if showEach:
                        print("Driver: ", driver)
                        print("State: ", driver_states[state])
                        print("Number of raw images: ", num_img)
                        print("Mean image shape: ", each_mean_img.shape)
                        import matplotlib.pyplot as plt
                        plt.imshow(each_mean_img / 255.)
                        plt.show()
                        
                    image_info['state'].append(state)
                    image_info['driver'].append(driver)
                    image_info['path'].append(path)
                    image_info['frames'].append(frames)
                    
        if not byDriver: imgs, num_imgs = {}, {}
        for state in self.states:
            if not byDriver: imgs[state], num_imgs = np.zeros((720, 1280, 3)), 0
                
            for driver in self.drivers:
                if byDriver:
                    print("Driver: ", driver)
                    print("State: ", driver_states[state])
                    print("Number of raw images: ", num_mean_imgs[state][driver])
                    print("Mean image shape: ", mean_img[state][driver].shape)
                    import matplotlib.pyplot as plt
                    plt.imshow(mean_img[state][driver] / 255.)
                    plt.show()
                    
                else:
                    if imgs[state].sum() == 0:
                        imgs[state] = mean_img[state][driver]
                        num_imgs[state] += num_mean_imgs[state][driver]
                    else:
                        imgs[state] = ((imgs[state] + mean_img[state][driver]) / 2.)
                        num_imgs[state] += num_mean_imgs[state][driver]
                
            if not byDriver:
                print("State: ", driver_states[state])
                print("Mean image shape: ", imgs[state].shape)
                print("Number of raw images: ", num_imgs[state])
                import matplotlib.pyplot as plt
                plt.imshow(imgs[state] / 255.)
                plt.show()
                
        return mean_img, image_info
    
    def show_mean_image_draft(self, mean_img, byDriver=True, showEach=False):
        if not byDriver: imgs = {}
        for state in self.states:
            if not byDriver: imgs[state] = np.zeros((720, 1280, 3))
                
            for driver in self.drivers:
                if byDriver:
                    print("Driver: ", driver)
                    print("State: ", driver_states[state])
                    print("Mean image shape: ", mean_img[state][driver].shape)
                    import matplotlib.pyplot as plt
                    plt.imshow(mean_img[state][driver] / 255.)
                    plt.show()
                    
                else:
                    if imgs[state].sum() == 0:
                        imgs[state] = mean_img[state][driver]
                    else:
                        imgs[state] = ((imgs[state] + mean_img[state][driver]) / 2.)
                
            if not byDriver:
                print("State: ", driver_states[state])
                print("Mean image shape: ", imgs[state].shape)
                import matplotlib.pyplot as plt
                plt.imshow(imgs[state] / 255.)
                plt.show()
        return mean_img, None


class DataLoader(DataAnalyzer):
    '''
    A data loader for Driver Monitoring System Initialize the basic data to be loaded.
    Args: 
        drivers: `list` or `tuple`, the registered name of meta data. 
        Loads data only for passed driver(s).

        states: `list` or `tuple`, the predefined driver's states.
        Loads data only for passed state(s).
        Reference: "Multimodal data collection system for driver emotion recognition based on self-reporting in real-world driving." Oh, Geesung, et al., 2022, Sensors.

        name: `str`, optional
        the name of the data loader.
        
        kwargs : mapping, optional
        a dictionary of keyword arguments passed into func.

    ```python
    from utils import DataLoader

    data_loader = DataLoader()
    ```
    Once the data loader is created, you can get the data
    with `data_loader.get_data()`.
    '''
    
    def __init__(self, drivers=[], states=[], batch_size=32, shuffle=True, name=None, **kwargs):
        super().__init__(self, drivers, states, name)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.kwargs = kwargs
        
    def get_y(self):
        print(self.HMI_data)
                
        
        
    def get_series_x(self, cols, prev_sec):
        """get the total input data of series data such as CAN, BIO data with adjusting sampling time.
        
        Args:
            cols: list of tuple, 
            input data features of series data.
            prev_sec: `integer` or `float`, 
            time range to observe until state is self-reported.
            byDriver: `bool`, default:True
                If True, plot the bat chart for each driver.

        retern:
            `DataFrame`, the mean value of a given column by state(s) and driver(s).
        """
        col = str(col)

        if col in cols_BIO:
            if col == 'IBI': raise ValueError("{} is not supported yet.".format(col))
            df = self.load_bio_data(col)
        
        elif col in cols_CAN_conti:
            df = self.load_can_data(col)
        
        else:
            raise ValueError("Invalid argument:", col)
        
        show_dict = {
            'state':[],
            'driver':[],
            'count':[],
            'mean_value':[],
        }
        
        condi_state = (lambda state: self.HMI_data['state']==state)
        condi_driver = (lambda driver: self.HMI_data['driver']==driver)
        
        condi_time = (lambda timestamp: df['timestamp']<=timestamp)
        condi_time_prev = (lambda timestamp: df['timestamp']>=timestamp-prev_sec)
        
    def __call__(self, idx, mode):
        '''
        """
        This module is compatible with Python versions up to and including 0.2.
        """

        :param idx:
        :param mode:
        :return:
        '''

        # return batch_x, batch_y

if __name__ == "__main__":

    # states = ['Angry/Disgusting', 'Sad/Fatigued', 'Unknown']
    # states = ['Happy/Neutral', 'Excited/Surprised', 'Unknown']
    # data_loader = DataLoader(states=states)
    data_loader = DataLoader()
    # help(data_loader)
    # print(data_loader.drivers)
    # print(data_loader.states)
    # print(data_loader.meta_data)
    print(data_loader.HMI_data)
    data_loader.draw_barplot()
    # df = data_loader.meta_info
    # print(df[df['driver'].isin(['GeesungOh'])])


