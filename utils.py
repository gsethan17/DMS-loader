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

    def __init__(self, drivers=[], states=[], name=None):

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
        """plot the number of self-reported driver's states as a bar chart
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
    
    def plot_mean_value(self, col, deltaT, byDriver=True):
        """plot the mean of mean value of a given column for a period time until each state is reported as a bar chart.
        
        Args:
            col: `str`, data feature.
            
            deltaT: `list`, ex) [{start}, {end}],
            Time difference from self-reporting time to start and end for time range.
            
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
        
        condi_time = (lambda timestamp: df['timestamp']<=timestamp+deltaT[1])
        condi_time_prev = (lambda timestamp: df['timestamp']>=timestamp+deltaT[0])
        
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
            
    
    def show_mean_image(self, view, mode, deltaT, byDriver=True, showEach=False):
        """show the mean of mean image of a given view and mode camera for a period time until each state is reported.
        
        Args:
            view: `str`, 'front' or 'side', position of camera installation.
            
            mode: `str`, 'color' or 'ir', RGB camera of IR camera.
            
            deltaT: `list`, ex) [{start}, {end}],
            Time difference from self-reporting time to start and end 
            for time range.
            
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
        
        condi_time = (lambda timestamp: df['timestamp']<=timestamp+deltaT[1])
        condi_time_prev = (lambda timestamp: df['timestamp']>=timestamp+deltaT[0])
        
        for state in self.states:
            for driver in self.drivers:
                time_list = self.HMI_data[condi_state(state) & condi_driver(driver)]['time']
                for timestamp in time_list:
                    # print(timestamp+deltaT[0])
                    # print(timestamp+deltaT[1])
                    paths = df[condi_time(timestamp) & condi_time_prev(timestamp)]['dir_path']
                    # print(paths)
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

from tensorflow.keras.utils import Sequence
class DataLoader(DataAnalyzer, Sequence):
    '''A data loader for Driver Monitoring System Initialize the basic data to be loaded.
    Args: 
        drivers: `list` or `tuple`, the registered name of meta data. 
        Loads data only for passed driver(s).

        states: `list` or `tuple`, the predefined driver's states.
        Loads data only for passed state(s).
        Reference: "Multimodal data collection system for driver emotion recognition based on self-reporting in real-world driving." Oh, Geesung, et al., 2022, Sensors.
        
        batch_size: `int`,
        
        shuffle: `bool`,

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
        super().__init__(drivers, states, name)
        
        self.convert_y_index = {x:i for i, x in enumerate(self.states)}
        self.convert_label = {i:driver_states[x] for i, x in enumerate(self.states)}
        self.i_mat = np.eye(len(self.states))
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.kwargs = kwargs['kwargs']
        
        self.df = {}
        for col_bio in self.kwargs['cols_BIO']:
            self.df[col_bio] = df = self.load_bio_data(col_bio)
        
        for col_can in self.kwargs['cols_CAN']:
            self.df[col_can] = df = self.load_can_data(col_can)
            
        self.df['image'] = self.load_video_info()
        
        
        self.condi_time = {}
        self.condi_time['CAN'] = (lambda col, timestamp: self.df[col]['timestamp']<=timestamp)
        self.condi_time['BIO'] = (lambda col, timestamp: self.df[col]['timestamp']<=timestamp)
        self.condi_time['image'] = (lambda timestamp: self.df['image']['timestamp']<=timestamp)
        
        self.condi_time_prev = {}
        self.condi_time_prev['CAN'] = (lambda col, timestamp: self.df[col]['timestamp']>=timestamp)
        self.condi_time_prev['BIO'] = (lambda col, timestamp: self.df[col]['timestamp']>=timestamp)
        self.condi_time_prev['image'] = (lambda timestamp: self.df['image']['timestamp']>=timestamp)
        
        self.dt = {}
        self.dt['CAN'] = self.kwargs['deltaT_CAN'][1] - self.kwargs['deltaT_CAN'][0]
        self.dt['BIO'] = self.kwargs['deltaT_BIO'][1] - self.kwargs['deltaT_BIO'][0]
        
        self.timestep = {}
        self.timestep['CAN'] = int((self.dt['CAN'] / self.kwargs['sampling_time_CAN']) - 1)
        self.timestep['BIO'] = int((self.dt['BIO'] / self.kwargs['sampling_time_BIO']) - 1)
        
        self.on_epoch_end()
        
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.HMI_data))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
            
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size] 
        
        batch_onehot_ys, batch_timestamps = self.get_y(indices)
        
        
        
        batch_x_can, batch_x_bio, batch_x_images, except_times = self.get_x(batch_timestamps)
        
        del_indices = [list(batch_timestamps).index(e) for e in except_times]
        if len(del_indices) > 0:
            batch_onehot_ys = np.delete(batch_onehot_ys, del_indices, 0)
        return (batch_x_can, batch_x_bio, batch_x_images), batch_onehot_ys
        
        
    def get_y(self, indices):
        
        timestamps = self.HMI_data.loc[indices, 'time'].values
        ys = [self.convert_y_index[x] for x in self.HMI_data.loc[indices, 'state'].values]
        onehot_ys = self.i_mat[ys]
        
        return onehot_ys, timestamps
                
    
    
    def get_x_serial(self, timestamp, stream):
        
        std_time = timestamp+self.kwargs['deltaT_{}'.format(stream)][0]
        end_time = timestamp+self.kwargs['deltaT_{}'.format(stream)][1]
        time_stamp = np.arange(std_time, end_time, self.kwargs['sampling_time_{}'.format(stream)])
        
        
        x_sample = np.array([])
        for i, col in enumerate(self.kwargs['cols_{}'.format(stream)]):
            xs = []
            for t in range(len(time_stamp)-1):
                x = self.df[col][self.condi_time[stream](col, time_stamp[t+1]) & self.condi_time_prev[stream](col, time_stamp[t])][col].mean()
                
                xs.append(x)
            
            # if len(xs) < self.timestep_can:
                # return None    
            xs = np.array(xs).reshape(-1,1)
            
            if i == 0:
                x_sample = xs[:self.timestep[stream]]
            else:
                x_sample = np.concatenate((x_sample, xs[:self.timestep[stream]]), axis=-1)
            
        return np.expand_dims(x_sample, axis=0)
            
    def get_x_image(self, timestamp):
        self.condi_time['image'] = (lambda timestamp: self.df['image']['timestamp']<=timestamp)
        self.condi_time_prev['image'] = (lambda timestamp: self.df['image']['timestamp']>=timestamp)
        
        std_time = timestamp+self.kwargs['deltaT_image'][0]
        end_time = timestamp+self.kwargs['deltaT_image'][1]
        time_stamp = np.arange(std_time, end_time, 1/self.kwargs['sampling_time_image'])
        
        x_image = {}
        
        for t in range(len(time_stamp)-1):
            paths = self.df['image'][self.condi_time['image'](time_stamp[t+1]) & self.condi_time_prev['image'](time_stamp[t])]['dir_path']
            frames = self.df['image'][self.condi_time['image'](time_stamp[t+1]) & self.condi_time_prev['image'](time_stamp[t])]['frame']
            
            if len(paths) == 0:
                # raise ValueError("Image sampling time may be higher than camera fps.")
                return False, None
                        
            path = list(set(paths))[0]
            
            for i, view in enumerate(self.kwargs['image_view']):
                mode = self.kwargs['image_mode'][i]
                
                imgs = self.get_image(path, frames, view, mode)
                
                if t == 0:
                    x_image[view] = np.array(imgs)[:1] / 255.
                else:
                    x_image[view] = np.concatenate((x_image[view], np.array(imgs)[:1] / 255.), axis=0)
                    
        return True, x_image
    
    
    def get_x(self, timestamps):
        batch_x_can = np.array([])
        batch_x_bio = np.array([])
        batch_x_images = {}
        for view in self.kwargs['image_view']:
            batch_x_images[view] = np.array([])
        
        except_time = []
        
        for i, timestamp in enumerate(timestamps):
            
            x_can_sample = self.get_x_serial(timestamp, 'CAN')
            if np.isnan(x_can_sample).sum() > 0:
                except_time.append(timestamp)
                continue
                
            x_bio_sample = self.get_x_serial(timestamp, 'BIO')
            if np.isnan(x_bio_sample).sum() > 0:
                except_time.append(timestamp)
                continue
            
            flag_img, x_image_samples = self.get_x_image(timestamp)
            
            if not flag_img:
                except_time.append(timestamp)
                continue
            
            if len(batch_x_can) == 0:
                batch_x_can = x_can_sample
                batch_x_bio = x_bio_sample
                for view in self.kwargs['image_view']:
                    batch_x_images[view] = np.expand_dims(x_image_samples[view], axis=0)
            else:
                batch_x_can = np.concatenate((batch_x_can, x_can_sample), axis=0)
                batch_x_bio = np.concatenate((batch_x_bio, x_bio_sample), axis=0)
                for view in self.kwargs['image_view']:
                    batch_x_images[view] = np.concatenate((batch_x_images[view], np.expand_dims(x_image_samples[view], axis=0)), axis=0)
                
        return batch_x_can, batch_x_bio, batch_x_images, except_time
                
            
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


