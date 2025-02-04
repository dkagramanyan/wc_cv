import os
import requests
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

from lmfit.models import Model

from sklearn.cluster import KMeans

from shapely.geometry import Polygon

from radio_beam.commonbeam import getMinVolEllipse

from scipy import ndimage as ndi
from scipy.spatial import distance

from skimage import io, color, filters, morphology, util
from skimage.measure import EllipseModel
from skimage.color import rgb2gray
from skimage import filters, util
from skimage.morphology import disk, skeletonize, ball
from skimage.measure import approximate_polygon
from skimage import transform

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from sklearn.linear_model import LinearRegression

from scipy import ndimage
import os
from bresenham import bresenham
from shapely.geometry import Polygon, LineString
import copy
import cv2
from tqdm.notebook import tqdm

from scipy.spatial import ConvexHull

import sys
import logging
import time
import glob
from logging import StreamHandler, Formatter

import json
import networkx as nx

from collections import Counter

from crdp import rdp
from pathlib import Path
from torch.utils.data import Dataset

from multiprocessing import Lock, Process, Queue, current_process
import multiprocessing

handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(handler)

file_path = os.getcwd() + '/utils.py'

# median filter +
# otsu - 
# sobel + 
# canny +

class Crack():
    @classmethod
    def align_figures(cls, orig_img_padded, tol):
        cnts = grainMark.get_contours(orig_img_padded,tol=tol)
        cnts = [np.array(cnt)[:-1] for cnt in cnts if len(cnt)>2]
        cnts_adj = [cnt.reshape((-1,1,2)) for cnt in cnts ]
        
        white_img = np.full((orig_img_padded.shape[0],orig_img_padded.shape[1],3),255)
        white_img = np.ascontiguousarray(white_img, dtype=np.uint8)
        img_viz = cv2.drawContours(white_img,cnts_adj,-1,(127,127,127),-1).astype(np.uint8)
        
        return img_viz, cnts
        
    @classmethod
    def preprocess_graph_image(cls, image, r=2, border = 30, border_node_eps=10, tol = 5):
    
        border_eps = border + border_node_eps
    
        # too small disk causes wrong contours detection
        # check double median filter
        tmp_img = SEMDataset.preprocess_image(image, pad=True, border=border,disk=8)
        
        img_aligned,cnts = cls.align_figures(tmp_img,tol)
        
        img_shape=np.array(img_aligned.shape)
        
        # coord2index
        image_nodes_coord2nodes_index={}
        nodes_index2global_nodes_coord={}
        nodes_index2global_contour_index={}
        nodes_index2local_contour_index={}
        num_of_nodes=0
        
        for i,points in enumerate(reversed(cnts)):
            for j, point in enumerate(points):
                x,y = point[0],point[1]
                image_nodes_coord2nodes_index[(y,x)]=num_of_nodes
                nodes_index2global_nodes_coord[num_of_nodes]=(y,x)
                nodes_index2global_contour_index[num_of_nodes] = i
                nodes_index2local_contour_index[num_of_nodes] = j 
                num_of_nodes+=1
        
        # entry points
        entry_nodes=[]
        
        entry_dict={}
        y_entry_max=0
        
        for points in cnts:
            for point in points:
                if point[1]<border_eps:
                    x,y = point[0],point[1]
                    # condition to make end exit poits below start points 
                    if y_entry_max<y:
                        y_entry_max=y
                    
                    index=image_nodes_coord2nodes_index[(y,x)]
                    entry_dict[index]=1
                    entry_nodes.append(index)
        
        # exit points
        exit_nodes=[]
        exit_dict={}
        
        for points in cnts:
            for point in points:
                if (img_shape[0] - point[1] < border_eps):
                    x,y = point[0],point[1]
                    index=image_nodes_coord2nodes_index[(y,x)]
                    exit_nodes.append(index)
                    exit_dict[index]=1
        
        img_drawings = copy.copy(Image.fromarray(img_aligned)).convert('RGB')
        img_drawings = grainDraw.draw_contours(img_drawings, cnts=cnts, color_corner=(0, 139, 139), color_line = (255, 140, 0),corners = True)
        draw = ImageDraw.Draw(img_drawings)
        
        # entry blue
        for key in entry_nodes:
            x,y=nodes_index2global_nodes_coord[key]
            draw.ellipse((y - r, x - r, y + r, x + r), fill=(0,0,255), width=1)
            
        # exit red
        for key in exit_nodes:
            x,y=nodes_index2global_nodes_coord[key]
            draw.ellipse((y - r, x - r, y + r, x + r), fill=(255,0,0), width=1)
            
        return entry_nodes, exit_nodes, num_of_nodes, img_aligned, img_drawings, nodes_index2global_nodes_coord, cnts, nodes_index2global_contour_index, nodes_index2local_contour_index, image_nodes_coord2nodes_index

    @classmethod
    def create_crack_graph(cls,
                           img_shape,
                           num_of_nodes,
                           nodes_index2global_nodes_coord,
                           cnts,
                           eps=100,
                           line_eps=3,
                           border = 30,
                           border_eps = 0,
                           border_number_min = 2,
                           border_pixel=255,
                           same_node_eps = 5):
        
        g = nx.DiGraph()
        image_node_coord2node_index = np.zeros(img_shape,dtype=np.int32)
        for key in range(num_of_nodes):
            x,y=nodes_index2global_nodes_coord[key]
            image_node_coord2node_index[x,y]=key
            g.add_node(key, pos=(y,x)) 
        
        m=[]
        a=[]
        
        
        img_contours = grainDraw.draw_contours(Image.fromarray(np.zeros((img_shape[0],img_shape[1]))), color_line = (255), cnts=cnts,corners = False)
        img_contours_np = np.array(img_contours)
        
        for start_node_index in tqdm(range(num_of_nodes)):
            
            # choose cell
            start_node_x,start_node_y=nodes_index2global_nodes_coord[start_node_index]
            
            # for rectangular vertical slice only!
            ###############################################
    
            # left y slice border
            if start_node_y-eps<0:
                left_border_y=0
            else:
                left_border_y=start_node_y-eps
        
            # right y slice border
            if start_node_y+eps>image_node_coord2node_index.shape[1]:
                right_border_y=image_node_coord2node_index.shape[1]-1
            else:
                right_border_y=start_node_y+eps
        
            # upper_border
            if start_node_x+eps>image_node_coord2node_index.shape[0]-1:
                upper_border=image_node_coord2node_index.shape[0]-1
            else:
                upper_border=start_node_x+eps
                
            map_slice = image_node_coord2node_index[start_node_x+1:upper_border,left_border_y:right_border_y]
            ###############################################
            
            nodes_indices_indices = np.where(map_slice.flatten()!=0)
            nodes_indices = map_slice.flatten()[nodes_indices_indices]
            
            # next node search
            for node_index in nodes_indices:
            
                end_node_x, end_node_y = nodes_index2global_nodes_coord[node_index]
        
                if abs(end_node_x-start_node_x)>same_node_eps or abs(end_node_y-start_node_y)>same_node_eps:
                
                    ab = LineString([(start_node_x, start_node_y), (end_node_x, end_node_y)])
                    left = ab.parallel_offset(line_eps, 'left')
                    left_p, _ = np.array(left.coords)
                    perp_v = np.array((start_node_x-left_p[0],start_node_y-left_p[1]))
                    perp_v = perp_v/np.linalg.norm(perp_v)
                    
                    mean_border_pixels=0
        
                        
                    for p in range(0 - line_eps, 1 + line_eps):
                        line_coords=np.array(list(bresenham(np.round(start_node_x+p*perp_v[0]).astype(np.int32),
                                                            np.round(start_node_y+p*perp_v[1]).astype(np.int32),
                                                            np.round(end_node_x+p*perp_v[0]).astype(np.int32),
                                                            np.round(end_node_y+p*perp_v[1]).astype(np.int32)
                                                           )))
                        
                        line_coords_pixels=img_contours_np[line_coords[:,0],line_coords[:,1]][2:-2]
                        border_pixels_num = np.where(line_coords_pixels==border_pixel)[0].shape[0]
                        if border_pixels_num<=border_eps:
                            mean_border_pixels+=1
        
                    if mean_border_pixels>=border_number_min and start_node_index!=node_index:
        
                        g.add_edge(start_node_index,node_index, weight=np.linalg.norm((end_node_x-start_node_x, end_node_y-start_node_y)))
    
    
        return g, img_contours

    @classmethod
    def find_intersection_2d(cls,p1, p2, p3, p4):
        """
        Check if two line segments defined by points (p1, p2) and (p3, p4) intersect.
        """
        # Vector representation
        v1 = p2 - p1
        v2 = p4 - p3
        
        # Cross product to check if lines are parallel
        cross = np.cross(v1, v2)
        
        if np.isclose(cross, 0):
            # Lines are parallel, check if they are collinear
            cross2 = np.cross(p3 - p1, v1)
            if np.isclose(cross2, 0):
                # Lines are collinear, check if they overlap
                t0 = np.dot(p3 - p1, v1) / np.dot(v1, v1)
                t1 = np.dot(p4 - p1, v1) / np.dot(v1, v1)
                if max(t0, t1) >= 0 and min(t0, t1) <= 1:
                    return True  # Lines overlap
            return False  # Lines are parallel but not collinear
        
        # Calculate the intersection point
        t = np.cross(p3 - p1, v2) / cross
        u = np.cross(p3 - p1, v1) / cross
        
        # Check if the intersection point lies within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True  # Lines intersect
        
        return False  # Lines do not intersect
        
    @classmethod
    def get_edge_type(cls,
                      node1,
                      node2,
                      cnts,
                      nodes_index2global_contour_index,
                      nodes_index2global_nodes_coord,
                      image_nodes_coord2nodes_index,
                      nodes_index2local_contour_index):
        
        cnt_index_1 = nodes_index2global_contour_index[node1]
        cnt_index_2 = nodes_index2global_contour_index[node2]
    
        edge_type=0
        
        # different contours, WC-WC
        if cnt_index_1!=cnt_index_2:
            edge_type=0
            
        # same contour, WC-Co 
        elif abs(node1-node2)<2:
            edge_type=1
            
        else:
    
            x1,y1 = nodes_index2global_nodes_coord[node1]
            x2,y2 = nodes_index2global_nodes_coord[node2]
            
            cnt = cnts[-cnt_index_1-1]
            cnt_point_1_index = nodes_index2local_contour_index[node1]
            
            node0_x,node0_y=cnt[cnt_point_1_index-1]
            new_key=cnt_point_1_index+1
            if new_key>=len(cnt):
                new_key=new_key-len(cnt)
            node3_x,node3_y=cnt[new_key]            
    
            node0=image_nodes_coord2nodes_index[(node0_y,node0_x)]
            node3=image_nodes_coord2nodes_index[(node3_y,node3_x)]
    
            if len(np.intersect1d([node1,node2],[node0,node3]))>0:
                edge_type=1
            else:
                x0,y0 = nodes_index2global_nodes_coord[node0]
                x3,y3 = nodes_index2global_nodes_coord[node3]
                
                inter = cls.find_intersection_2d(np.array((x1,y1)),
                                             np.array((x2,y2)),
                                             np.array((x0,y0)),
                                             np.array((x3,y3)))
                #  has intersection, Co
                if inter:
                    edge_type=2
                else:
                    # no intersection, WC 
                    edge_type=0
    
        return edge_type
    


class SEMDataset(Dataset):
    def __init__(self, images_folder_path, no_cache=False, max_images_num_per_class=10, workers=None):
        
        self.cached_dir = 'tmp/'+ images_folder_path.split('/')[-1]
        
        if workers is None:
            workers = multiprocessing.cpu_count()-1
        
        if images_folder_path[-1]=='/':
            raise ValueError('remove last "/" in path ')
        
        if os.path.exists(self.cached_dir) is False or no_cache:
            
            Path(self.cached_dir).mkdir(parents=True, exist_ok=True)
            
            folders_paths = glob.glob(images_folder_path + '/*')
            images_paths = []
            
            for folder_path in folders_paths:
                images_paths.extend(glob.glob(folder_path + '/*')[:max_images_num_per_class])
                folder_name = folder_path.split('/')[-1]
                
                new_folder_path = self.cached_dir + '/' + folder_name
                Path(new_folder_path).mkdir(parents=True, exist_ok=True)
            
            number_of_tasks = len(folders_paths)
            images_paths = np.array(images_paths).reshape((number_of_tasks,-1))

            tasks_to_accomplish = Queue()
            tqdm_queue = Queue()
            processes = []
        
            for i in range(number_of_tasks):
                tasks_to_accomplish.put(images_paths[i])
            
            for w in range(workers):
                p = Process(target=self.do_job, args=(tasks_to_accomplish,self.cached_dir,tqdm_queue))
                processes.append(p)
                p.start()
                
            total = images_paths.shape[0]*images_paths.shape[1]
                
            with tqdm(total=total) as pbar:
                completed = 0
                while completed < total:
                    tqdm_queue.get()
                    pbar.update(1)
                    completed += 1

            for p in processes:
                p.join()
            
        folders_paths = glob.glob(self.cached_dir + '/*')
        folder_names= [folder_path.split('/')[-1] for folder_path in folders_paths] 
            
        self.images_paths = np.array([glob.glob(self.cached_dir + f'/{folder_name}/*')[:max_images_num_per_class] for folder_name in folder_names])
    
    @classmethod
    def do_job(cls, tasks_to_accomplish, cached_dir, tqdm_queue):
        while not tasks_to_accomplish.empty():
            images_paths = tasks_to_accomplish.get()
            for image_path in images_paths:
                image = io.imread(image_path)
                image = cls.preprocess_image(image)

                splitted=image_path.split('/')
                folder_name, file_name = splitted[-2], splitted[-1]
                file_name = file_name.split('.')[0]

                io.imsave(cached_dir + '/' + folder_name + '/' + file_name + '.png', image)
                tqdm_queue.put(1)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, class_idx, idx):
        path = self.images_paths[class_idx, idx]
        image = io.imread(path)
        return image, path

    @classmethod
    def pad_with(cls, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 255)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    @classmethod
    def preprocess_image(cls, image, pad=False, border=30,disk=3):
        if len(image.shape)==3:
            image = color.rgb2gray(image)
        
        image = filters.rank.median(image, morphology.disk(disk))

        global_thresh = filters.threshold_otsu(image)
        image = image > global_thresh
        binary = image*255
        binary = binary.astype(np.uint8)

        if pad:
            binary = np.pad(binary, border, cls.pad_with)

        grad = abs(filters.rank.gradient(binary, morphology.disk(1)))
        bin_grad = (1 - binary + grad) * 127
        bin_grad = np.clip(bin_grad, 0, 255).astype(np.uint8)

        return bin_grad
        

class grainPreprocess():

    @classmethod
    def image_preprocess_kmeans(cls, image: np.ndarray, h=135, k=1, n_clusters=3, pos=1) -> np.ndarray:
        """
        :param image: array (height,width,channels)
        :param h: int scalar
        :param k: float scalar
        :param n_clusters: int scalar
        :param pos: int scalar, cluster index
        :return: ndarray (height,width)
        """
        #
        # выделение границ при помощи кластеризации
        # и выравнивание шума медианным фильтром
        # pos отвечает за выбор кластера, который будет отображен на возвращенном изображении
        #
        combined = cls.combine(image, h, k)

        clustered, colors = grainMorphology.kmeans_image(combined, n_clusters)
        cluster = clustered == colors[pos]
        cluster = np.array(cluster * 255, dtype='uint8')

        new_image = filters.median(cluster, disk(2))
        return new_image

    @classmethod
    def imdivide(cls, image: np.ndarray, h: int, side: str) -> np.ndarray:
        """
        :param image: ndarray (height,width,channels)
        :param h: int scalar
        :param side: str 'left'
        :return: ndarray (height,width/2,channels)
        """
        #
        # возвращает левую или правую часть полученного изображения
        #
        height, width = image.shape
        sides = {'left': 0, 'right': 1}
        shapes = [(0, height - h, 0, width // 2), (0, height - h, width // 2, width)]
        shape = shapes[sides[side]]

        return image[shape[0]:shape[1], shape[2]:shape[3]]

    @classmethod
    def combine(cls, image: np.ndarray, h: int, k=0.5) -> np.ndarray:
        """
        :param image: ndarray (height,width,channels)
        :param h: int scalar
        :param k: float scalar
        :return: ndarray (height,width/2,channels)
        """
        #
        #  накладывает левую и правые части изображения
        #  если k=1, то на выходе будет левая часть изображения, если k=0, то будет правая часть
        #
        left_img = cls.imdivide(image, h, 'left')
        right_img = cls.imdivide(image, h, 'right')

        l = k
        r = 1 - l
        gray = np.array(left_img) * l
        gray += np.array(right_img) * r

        return gray.astype('uint8')

    @classmethod
    def do_otsu(cls, img: np.ndarray) -> np.ndarray:
        """
        :param img: ndarray (height,width,channels)
        :return: ndarray (height,width), Boolean
        """
        #
        # бинаризация отсу
        #
        global_thresh = filters.threshold_otsu(img)
        binary_global = img > global_thresh

        return binary_global.astype('uint8')

    @classmethod
    def image_preprocess(cls, image: np.ndarray) -> np.ndarray:
        """
        :param image: ndarray (height,width,channels)
        :return: ndarray (height,width,1)
        """
        #
        # комбинация медианного фильтра, биноризации и градиента
        # у зерен значение пикселя - 0, у регионов связ. в-ва - 127,а у их границы - 254
        #
        unsigned_image = util.img_as_ubyte(image)
        if len(unsigned_image.shape) < 3:
            unsigned_image = unsigned_image[..., np.newaxis]
        denoised = filters.rank.median(unsigned_image, ball(3))
        binary = cls.do_otsu(denoised)
        grad = abs(filters.rank.gradient(binary, ball(1)))
        bin_grad = (1 - binary + grad) * 127

        return bin_grad.astype(np.uint8)

    @classmethod
    def read_preprocess_data(cls,
                             images_dir,
                             max_images_num_per_class=100,
                             save=False,
                             crop_bottom=False,
                             h=135, resize_shape=None,
                             preprocess_transform=None,
                             save_name='all_images'
                             ):
        """
        :param images_dir: str
        :param max_images_num_per_class: int
        :param preprocess: Bool
        :param save: Bool
        :param crop_bottom: Bool
        :param h: int
        :param resize: Bool
        :param resize_shape: tuple (width, height, channels)
        :param save_name: str
        :return: ndarray (n_classes, n_images_per_class, width, height, channels)
        """

        folders_names = glob.glob(images_dir + '*')
        images_paths = [glob.glob(folder_name + '/*')[:max_images_num_per_class] for folder_name in folders_names]

        preproc_images = []
        if len(images_paths) > 0:
            if preprocess_transform is None:
                preprocess_transform = [grainPreprocess.image_preprocess]
            elif preprocess_transform is False:
                preprocess_transform = None

            for i, images_list_paths in enumerate(images_paths):
                preproc_images.append([])
                for image_path in tqdm(images_list_paths):
                    image = io.imread(image_path).astype(np.uint8)
                    # вырезает нижнюю полоску фотографии с линекой и тд
                    # !!!!!! убрать !!!!!
                    if crop_bottom:
                        image = grainPreprocess.combine(image, h)

                    # ресайзит изображения
                    if resize_shape is not None:
                        if resize_shape is not None:
                            image = transform.resize(image, resize_shape)
                        else:
                            print('No resize shape')

                    # последовательно применяет фильтры (медианный, отсу, собель и тд)
                    if preprocess_transform is not None:
                        for transf in preprocess_transform:
                            image = transf(image)

                    preproc_images[i].append(image)

            if save:
                np.save(f'{save_name}_images.npy', preproc_images)
                names_dict = dict((f'Class_{i}', name.replace('\\', '/')) for i, name in enumerate(folders_names))
                with open(f'{save_name}_metadata.json', 'w') as outfile:
                    json.dump(names_dict, outfile)

            return np.array(preproc_images), folders_names
        else:
            print('wrong images path')

    @classmethod
    def tiff2jpg(cls, folder_path, start_name=0, stop_name=-4, new_folder_path='resized'):
        """
        :param folder_path: str
        :param start_name: int
        :param stop_name: int
        :param new_folder_path: str
        :return: None
        """
        #
        # переводит из tiff 2^16 в jpg 2^8 бит
        #
        folders = os.listdir(folder_path)

        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)

        for folder in folders:
            if not os.path.exists(new_folder_path + '/' + folder):
                os.mkdir(new_folder_path + '/' + folder)

        for i, folder in enumerate(folders):
            images_names = os.listdir(folder_path + '/' + folder)
            for i, name in enumerate(images_names):
                if 'hdr' not in name:
                    img = io.imread(folder_path + '/' + folder + '/' + name)
                    img = (img / 255).astype('uint8')

                    io.imsave(new_folder_path + '/' + folder + '/' + name[start_name:stop_name] + '.jpg', img)

    @classmethod
    def get_example_images(cls, crop=True, preprocess=False):
        '''
        :return: ndarray [[img1],[img2]..]
        '''
        #
        # скачивает из контейнера s3 по 1 снимку каждого образца
        #

        images = []

        if crop:
            if preprocess:
                urls = CfgDataset.images_crop_preproc_urls
            else:
                urls = CfgDataset.images_crop_urls
        else:
            urls = CfgDataset.images_urls

        for url in urls:
            # logger.warning(f'downloading {url}')
            print(f'downloading {url}')
            file = requests.get(url, stream=True).raw
            img = np.asarray(Image.open(file))
            images.append(img)

        return np.array(images)


class grainMorphology():

    @classmethod
    def kmeans_image(cls, image, n_clusters=3):
        """
        :param image: ndarray (width, height, channels)
        :param n_clusters: int
        :return: (image, colors),colors - list of median colors of the clusters
        """
        #
        # кластеризует при помощи kmeans
        # и возвращает изображение с нанесенными цветами кластеров
        #
        img = image.copy()

        size = img.shape
        img = img.reshape(-1, 1)

        model = KMeans(n_clusters=n_clusters)
        clusters = model.fit_predict(img)

        colors = []
        for i in range(n_clusters):
            color = np.median(img[clusters == i])  # медианное значение пикселей у кластера
            img[clusters == i] = color
            colors.append(int(color))

        img = img.reshape(size)
        colors.sort()

        return img, colors


class grainFig():

    @classmethod
    def line(cls, point1, point2):
        """
        :param point1: tuple (int, int)
        :param point2: tuple (int, int)
        :return: ndarray (n_points,(x,y))
        """
        #
        # возвращает растровые координаты прямой между двумя точками при помощи алгоритма Брезенхема
        #
        line = []

        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        dx = x2 - x1
        dy = y2 - y1

        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

        if dx < 0: dx = -dx
        if dy < 0: dy = -dy

        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy

        x, y = x1, y1
        error, t = el / 2, 0

        line.append((x, y))

        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            line.append((x, y))
        return np.array(line).astype('int')

    @classmethod
    def rect(cls, point1, point2, r):
        """
        :param point1: tuple (int, int)
        :param point2: tuple (int, int)
        :param r: int
        :return: tuple (n_points, rect_diag*2,2 )
        """
        #
        # возвращает растровые координаты прямоугольника ширины 2r,
        # построеного между двумя точками 
        #
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        l1, l2 = (x2 - x1), (y2 - y1)

        l_len = (l1 ** 2 + l2 ** 2) ** 0.5
        l_len = int(l_len)

        a = (x1 - r * l2 / l_len), (y1 + r * l1 / l_len)
        b = (x1 + r * l2 / l_len), (y1 - r * l1 / l_len)

        side = cls.line(a, b)

        # a -> c
        # зачем умножать l_len на 2 ?
        lines = np.zeros((side.shape[0], l_len * 2, 2), dtype='int64')

        for i, left_point in enumerate(side):
            right_point = (left_point[0] + l1), (left_point[1] + l2)
            line_points = cls.line(left_point, right_point)
            for j, point in enumerate(line_points):
                lines[i, j] = point

        return lines


class grainMark():
    @classmethod
    def mark_corners_and_classes(cls, image, max_num=100000, sens=0.1, max_dist=1):
        """
        :param image: ndarray (width, height, channels)
        :param max_num: int
        :param sens: float
        :param max_dist: int
        :return: corners, classes, num
        """
        #
        # НЕТ ГАРАНТИИ РАБОТЫ
        # возвращает всевозможные координаты углов и исходное изображение с нанесенными классами клстеров градиента
        #
        corners = cv2.goodFeaturesToTrack(image, max_num, sens, max_dist)
        corners = np.int0(corners)
        x = copy.copy(corners[:, 0, 1])
        y = copy.copy(corners[:, 0, 0])
        corners[:, 0, 0], corners[:, 0, 1] = x, y

        classes = filters.rank.gradient(image, disk(1)) < 250
        classes, num = ndi.label(classes)
        return corners, classes, num

    @classmethod
    def mean_pixel(cls, image, point1, point2, r):
        """
        :param image: ndarray (width, height, channels)
        :param point1: tuple (int, int)
        :param point2: tuple (int, int)
        :param r: int
        :return: mean, dist
        """
        #
        # НЕТ ГАРАНТИИ РАБОТЫ
        # возвращает среднее значение пикселей прямоугольника ширины 2r, построеного между двумя точками
        #
        val2, num2 = cls.draw_rect(image, point2, point1, r)
        val = val1 + val2
        num = num1 + num2

        if num != 0 and val != 0:
            mean = (val / num) / 255
            dist = distance.euclidean(point1, point2)
        else:
            mean = 1
            dist = 1
        return mean, dist

    @classmethod
    def get_row_contours(cls, image):
        """
        :param image: ndarray (width, height,3)
        :return: list (N_contours, (M_points,2) )
        """
        #
        # Возвращает кооридинаты пикселей контуров каждого региона
        #
        edges = cv2.Canny(image, 0, 255, L2gradient=False)

        # направление обхода контура по часовой стрелке
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        new_contours = []
        for cnt in contours:
            new_contours.append(np.array(cnt).reshape((-1, 2)))
        return new_contours

    @classmethod
    def get_contours(cls, image, tol=3):
        """
        :param image: ndarray (width, height,3)
        :param tol: int Maximum distance from original points of polygon to approximated polygonal chain
        :return: list (N_contours, (M_points,2) )
        """
        #
        # уменьшение количества точек контура при помощи алгоритма Дугласа-Пекера
        #
        contours = cls.get_row_contours(image)

        new_contours = []
        for j, cnt in enumerate(contours):
            if len(cnt) > 2:
                # coords = approximate_polygon(cnt, tolerance=tol)
                coords = rdp(cnt, tol)
                new_contours.append(coords)
            else:
                continue

        return new_contours

    @classmethod
    def get_angles(cls, image, border_eps=5, tol=3):
        """
        :param image: ndarray (width, height,1), only preprocessed image
        :param thr: int, distance from original image edge to inner image edge (rect in rect)
        :return: angles ndarray shape (n)
        """
        #
        # Возвращает углы с направлением обхода контура против часовой стрелки, углы >180 градусов учитываются.
        # На вход принимает только обработанное изображение
        #

        cnts = cls.get_row_contours(image)
        angles = []

        for j, cnt in enumerate(cnts):
            rules = [image.shape[0] - cnt[:, 0].max() > border_eps, cnt[:, 0].min() > border_eps,
                     image.shape[1] - cnt[:, 1].max() > border_eps, cnt[:, 1].min() > border_eps]
            # проверяем находится ли контур у границы, но это срабатывает очень редко
            if all(rules):
                # cnt_aprox = approximate_polygon(cnt, tolerance=tol)
                cnt_aprox = rdp(cnt, tol)
                # основная проверка на неправильные контуры
                if len(cnt_aprox) > 3:
                    for i, point in enumerate(cnt_aprox[:-1]):

                        y1, x1 = cnt_aprox[i - 1]
                        y2, x2 = cnt_aprox[i]
                        y3, x3 = cnt_aprox[i + 1]

                        v1 = np.array((x1 - x2, y1 - y2)).reshape(1, 2)
                        v2 = np.array((x3 - x2, y3 - y2)).reshape(1, 2)

                        dot = np.dot(v1[0], v2[0])
                        dist1 = np.linalg.norm(v1[0])
                        dist2 = np.linalg.norm(v2[0])
                        cos = dot / (dist1 * dist2)

                        v = np.concatenate([v1, v2])
                        det = np.linalg.det(v)

                        if abs(cos) < 1:
                            ang = int(np.arccos(cos) * 180 / np.pi)

                            if det < 0:
                                angles.append(ang)
                            else:
                                angles.append(360 - ang)
                        else:
                            if det < 0:
                                angles.append(360)
                            else:
                                angles.append(0)

        return np.array(angles)

    @classmethod
    def get_mvee_params(cls, image, tol=0.2, debug=False):
        """
        :param image: ndarray (width, height,1), only preprocessed image
        :param tol: foat, koef of ellipse compactness
        :return: ndarray a_beams, b_beams, angles, centroids
        """
        #
        # возвращает полуоси и угол поворота фигуры minimal volume enclosing ellipsoid,
        # которая ограничивает исходные точки контура эллипсом. Для расчетов центр координатной оси
        # сдвигается на центроид полигона (исследуемого региона),
        # а затем сдвигается на среднее значение координат полигона
        # 
        raw_contours = grainMark.get_row_contours(image)
        a_beams = []
        b_beams = []
        angles = []
        centroids = []

        contours = []

        for i, cnt in enumerate(raw_contours):
            if len(cnt) > 2:
                try:
                    cnt = np.array(cnt)
                    polygon = Polygon(cnt)
                    contours.append(cnt)

                    x_centroid, y_centroid = polygon.centroid.coords[0]
                    points = cnt - (x_centroid, y_centroid)

                    x_norm, y_norm = points.mean(axis=0)
                    points = (points - (x_norm, y_norm))
                    data = getMinVolEllipse(points, tol)

                    xc, yc = data[0][0]
                    a, b = data[1]
                    sin = data[2][0][1]
                    angle = -np.arcsin(sin)

                    a_beams.append(a)
                    b_beams.append(b)
                    angles.append(angle)
                    centroids.append([x_centroid + x_norm, y_centroid + y_norm])
                except Exception:
                    if debug:
                        logger.warning(f'{file_path} error i={i}, singularity matrix error, no reason why',
                                       exc_info=debug)

        a_beams = np.array(a_beams, dtype='int32')
        b_beams = np.array(b_beams, dtype='int32')
        angles = np.array(angles, dtype='float32')
        centroids = np.array(centroids, dtype='int32')

        return a_beams, b_beams, angles, centroids, contours

    @classmethod
    def skeletons_coords(cls, image):
        """
        :param image: ndarray (width, height,1)
        :return: bones
        """
        #
        # на вход подается бинаризованное изображение
        # создает массив индивидуальных скелетов
        # пикселю скелета дается класс, на координатах которого он находится
        # координаты класса определяются ndi.label
        #
        skeleton = np.array(skeletonize(image))
        labels, classes_num = ndimage.label(image)

        bones = [[] for i in range(classes_num + 1)]

        for i in range(skeleton.shape[0]):
            for j in range(skeleton.shape[1]):
                if skeleton[i, j]:
                    label = labels[i, j]
                    bones[label].append((i, j))
        return bones


class grainShow():

    @classmethod
    def img_show(cls, image, N=20, cmap=plt.cm.nipy_spectral):
        """
        :param image:  ndarray (height,width,channels)
        :param N: int
        :param cmap: plt cmap
        :return: None
        """
        #
        # выводит изображение image
        #

        plt.figure(figsize=(N, N))
        plt.axis('off')
        plt.imshow(image, cmap=cmap)
        plt.show()

    @classmethod
    def enclosing_ellipse_show(cls, image, pos=0, tolerance=0.2, N=15):
        """
        :param image: ndarray (height,width,channels)
        :param pos: int
        :param tolerance: foat, koef of ellipse compactness
        :param N: int
        :return: None
        """
        #
        # Выводит точки многоугольника с позиции pos и описанного вокруг него эллипса
        #
        a_beams, b_beams, angles, cetroids, contours = grainMark.get_mvee_params(image, tolerance)
        approx = grainMark.get_row_contours(image)

        a = a_beams[pos]
        b = b_beams[pos]
        angle = angles[pos]
        print('полуось а ', a)
        print('полуось b ', b)
        print('угол поворота ', round(angle, 3), ' радиан')

        cnt = np.array(approx[pos])

        xp = cnt[:, 0]
        yp = cnt[:, 1]
        xc = cetroids[pos, 0]
        yc = cetroids[pos, 1]

        x, y = grainStats.ellipse(a, b, angle)

        plt.figure(figsize=(N, N))
        plt.plot(xp - xc, yp - yc)
        plt.scatter(0, 0)
        plt.plot(x, y)

        plt.show()
        
    
    @classmethod
    def angles_plot_base(cls, data, save_name, step, N, M,  save=False,indices=None, font_size=20,scatter_size=20):
        alloys_indices=range(len(data))

        if indices is not None:
            alloys_indices=indices

        plt.rcParams['font.size'] = '15'
        plt.figure(figsize=(N, M))

        # маркеры для одновременного вывода скаттера для разных классов
        # количество варкеров=количество классов-1
        markers = ['v', 's', 'D', 'd', 'p', '*','P']
        colors = ['orange', 'red','blue','green', 'indigo']

        legend=[]
        for i in alloys_indices:
            legend.append(data[i]['name']+' '+data[i]['type'])

        for i in alloys_indices:
            plt.plot(data[i]['gauss_approx_plot'][0], 
                     data[i]['gauss_approx_plot'][1],
                     color=colors[i])

        for i in alloys_indices:
            marker = markers[i]
            plt.scatter(data[i]['density_curve_scatter'][0],data[i]['density_curve_scatter'][1],  marker=marker,color=colors[i],s=scatter_size)

        plt.ylabel('p(x)', fontsize=font_size)
        plt.xlabel('углы, градусы', fontsize=font_size)

        x = [0,60,120,180,240,300,360]
        plt.xticks(x, x)

        plt.title(save_name)

        if save:
            plt.savefig(f'распределение_углов_{save_name}_шаг_{step}.png')

        plt.legend(legend)
        plt.show()
        
        for i in alloys_indices:
            print(data[i]['legend'])


class grainDraw():
    @classmethod
    def draw_corners(cls, image, corners, color=255):
        """
        :param image: ndarray (width, height, channels)
        :param corners: list (n_corners,2)
        :param color:  int
        :return: ndarray (width, height, channels)
        """
        #
        # НЕТ ГАРАНТИИ РАБОТЫ
        # Наносит на изображение точки в местах, где есть углы списка corners
        #
        image = copy.copy(image)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(image, (x, y), 3, color, -1)

        return image

    @classmethod
    def draw_contours(cls, image, cnts, color_corner=(0, 139, 139), color_line = (255, 140, 0),  r=2, e_width=5, l_width=2, corners = False):
    
        img = copy.copy(image)
        draw = ImageDraw.Draw(img)
    
        for j, cnt in enumerate(cnts):
            points_stack = np.stack([cnt, np.roll(cnt,shift=-1,axis=0)],axis=1)
            
            for points in points_stack:
                y1, x1 = points[0]
                y2, x2 = points[1]
                if corners:
                    draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=color_corner, width=e_width)
                draw.line((y1, x1, y2, x2), fill=color_line, width=l_width)
    
        return img
    


    @classmethod
    def draw_tree(cls, img, centres=False, leafs=False, nodes=False, bones=False):
        """
        :param img: ndarray (width, height)
        :param centres: Bool
        :param leafs: Bool
        :param nodes: Bool
        :param bones: Bool
        :return: ndarray (width, height, channels)
        """
        #
        # на вход подается бинаризованное изображение
        # рисует на инвертированном изображении скелет: точки их центров, листьев, узлов и пикселей скелета
        #

        image = img.copy() / 255

        skeleton = np.array(skeletonize(image)) * 255
        im = 1 - image + skeleton
        im = Image.fromarray(np.uint8(cm.gist_earth(im) * 255))
        draw = ImageDraw.Draw(im)

        if bones:
            for j, bone in enumerate(bones):
                for i, point in enumerate(bone):
                    x2, y2 = point
                    r = 1
                    draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=(89, 34, 0), width=5)

        if centres:
            for j, point in enumerate(centres):
                x2, y2 = point
                r = 2
                draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=(255, 0, 0), width=5)

        if leafs:
            for j, leaf in enumerate(leafs):
                for i, point in enumerate(leaf):
                    x2, y2 = point
                    r = 2
                    draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=(0, 255, 0), width=5)
        if nodes:
            for j, node in enumerate(nodes):
                for i, point in enumerate(node):
                    x2, y2 = point
                    r = 2
                    draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=(0, 0, 255), width=10)

        return np.array(im)


class grainStats():
    @classmethod
    def kernel_points(cls, image, point, step=1):
        """
        :param image: ndarray (width, height)
        :param point: tuple (2,)
        :param step: int
        :return: tuple (n_points,2)
        """
        #
        # возвращает координаты пикселей квадратной матрицы шириной 2*step, центр которой это point
        #
        x, y = point
        coords = []
        for xi in range(x - step, x + step + 1):
            for yi in range(y - step, y + step + 1):
                if xi < image.shape[0] and yi < image.shape[1]:
                    coords.append((xi, yi))
        return coords

    @classmethod
    def stats_preprocess(cls, array, step):
        """
        :param array: list, ndarray (n,)
        :param step: int
        :return: array_copy, array_copy_set, dens_curve
        """
        #
        # приведение углов к кратости, например 0,step,2*step и тд
        #
        if step != 0:
            array = np.array(array)
            val = array % step
            condition = val != 0

            adjusted_array = np.where(
                condition & (val < step / 2), 
                array - val, 
                np.where(condition, array + step - val, array)
            )

            new_array = np.round(adjusted_array)

            cnt = Counter(new_array)
            counts = np.array(list(cnt.items()), dtype=np.float32)
            counts = counts[counts[:, 0].argsort()]

            x = counts[:, 0]
            y = counts[:, 1]

            y = y/np.sum(y)
            return x, y
        
        else:
            print('step is 0, stats preprocess error')

    @classmethod
    def gaussian(cls, x, mu, sigma, amp=1):
        """
        :param x: list (n,)
        :param mu: float
        :param sigma: float
        :param amp: float
        :return: list (n,)
        """
        #
        # возвращает нормальную фунцию по заданным параметрам
        #
        return np.array((amp / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))

    @classmethod
    def gaussian_bimodal(cls, x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1):
        """
        :param x: list (n,)
        :param mu1: float
        :param mu2: float
        :param sigma1: float
        :param sigma2: float
        :param amp1: float
        :param amp2: float
        :return: list (n,)
        """
        #
        # возвращает бимодальную нормальную фунцию по заданным параметрам
        #
        return cls.gaussian(x, mu1, sigma1, amp1) + cls.gaussian(x, mu2, sigma2, amp2)

    @classmethod
    def gaussian_termodal(cls, x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1):
        """
        :param x: list (n,)
        :param mu1: float
        :param mu2: float
        :param mu3: float
        :param sigma1: float
        :param sigma2: float
        :param sigma3: float
        :param amp1: float
        :param amp2: float
        :param amp3: float
        :return: list (n,)
        """
        #
        # возвращает термодальную нормальную фунцию по заданным параметрам
        #
        return cls.gaussian(x, mu1, sigma1, amp1) + cls.gaussian(x, mu2, sigma2, amp2) + cls.gaussian(x, mu3, sigma3,
                                                                                                      amp3)

    @classmethod
    def ellipse(cls, a, b, angle, xc=0, yc=0, num=50):
        """
        :param a: float
        :param b: float
        :param angle: float, rad
        :param xc: float, center coord x
        :param yc: float, center coord y
        :param num: int, number of ellipse points
        :return: tuple (num, 2)
        """
        #
        #  возвращает координаты эллипса, построенного по заданным параметрам
        #  по умолчанию центр (0,0)
        #  угол в радианах, уменьшение угла обозначает поворот эллипса по часовой стрелке
        #
        xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, num),
                                       params=(xc, yc, a, b, angle))
        return xy[:, 0], xy[:, 1]


class grainApprox():

    @classmethod
    def gaussian_fit(cls, x, y, mu=1, sigma=1, amp=1):
        """
        :param x: list (n,)
        :param y: list (n,)
        :param mu: float
        :param sigma: float
        :param amp: float
        :return: mus, sigmas, amps
        """
        #
        # аппроксимация заданных точек нормальной функцией
        #
        gmodel = Model(grainStats.gaussian)
        res = gmodel.fit(y, x=x, mu=mu, sigma=sigma, amp=amp)

        mu = res.params['mu'].value
        sigma = res.params['sigma'].value
        amp = res.params['amp'].value

        return mu, sigma, amp

    @classmethod
    def gaussian_fit_bimodal(cls, x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1):
        """
        :param x: list (n,)
        :param y: list (n,)
        :param mu1: float
        :param mu2: float
        :param sigma1: float
        :param sigma2: float
        :param amp1: float
        :param amp2: float
        :return: mus, sigmas, amps
        """
        #
        # аппроксимация заданных точек бимодальной нормальной функцией
        #
        gmodel = Model(grainStats.gaussian_bimodal)
        res = gmodel.fit(y, x=x, mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, amp1=amp1, amp2=amp2)

        mus = [res.params['mu1'].value, res.params['mu2'].value]
        sigmas = [res.params['sigma1'].value, res.params['sigma2'].value]
        amps = [res.params['amp1'].value, res.params['amp2'].value]

        return mus, sigmas, amps

    @classmethod
    def gaussian_fit_termodal(cls, x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1,
                              amp3=1):
        """
        :param x: list (n,)
        :param y: list (n,)
        :param mu1: float
        :param mu2: float
        :param mu3: float
        :param sigma1: float
        :param sigma2: float
        :param sigma3: float
        :param amp1: float
        :param amp2: float
        :param amp3: float
        :return: mus, sigmas, amps
        """
        #
        # аппроксимация заданных точек термодальной нормальной функцией
        #
        gmodel = Model(grainStats.gaussian_termodal)
        res = gmodel.fit(y, x=x, mu1=mu1, mu2=mu2, mu3=mu3, sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, amp1=amp1,
                         amp2=amp2, amp3=amp3)

        mus = [res.params['mu1'].value, res.params['mu2'].value, res.params['mu3'].value]
        sigmas = [res.params['sigma1'].value, res.params['sigma2'].value, res.params['sigma3'].value]
        amps = [res.params['amp1'].value, res.params['amp2'].value, res.params['amp3'].value]

        return mus, sigmas, amps

    @classmethod
    def lin_regr_approx(cls, x, y):
        """
        :param x: list (n,1)
        :param y: list (n,1)
        :return: (x_pred, y_pred), k, b, angle, score
        """
        #
        # аппроксимация распределения линейной функцией и создание графика по параметрам распределения
        #

        x_pred = np.linspace(x.min(axis=0), x.max(axis=0), 50)

        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x_pred)

        k = reg.coef_[0][0]
        b = reg.predict([[0]])[0][0]

        angle = np.rad2deg(np.arctan(k))
        score = reg.score(x, y)

        return (x_pred, y_pred), k, b, angle, score

    @classmethod
    def bimodal_gauss_approx(cls, x, y):
        """
        :param x: list (n,)
        :param y: list (n,)
        :return: (x_gauss, y_gauss), mus, sigmas, amps
        """
        #
        # аппроксимация распределения бимодальным гауссом
        #

        mus, sigmas, amps = cls.gaussian_fit_bimodal(x, y)

        x_gauss = np.arange(0, 361)
        y_gauss = grainStats.gaussian_bimodal(x_gauss, mus[0], mus[1], sigmas[0], sigmas[1], amps[0], amps[1])

        return (x_gauss, y_gauss), mus, sigmas, amps


class grainGenerate():
    @classmethod
    def angles_legend(cls, images_amount, name, itype, step, mus, sigmas, amps, norm):
        """
        :param images_amount: int
        :param name: str
        :param itype: str
        :param step: int
        :param mus: float
        :param sigmas: float
        :param amps: float
        :param norm: int
        :return: str
        """
        #
        # создание легенды распределения углов
        #

        mu1 = round(mus[0], 2)
        sigma1 = round(sigmas[0], 2)
        amp1 = round(amps[0], 2)

        mu2 = round(mus[1], 2)
        sigma2 = round(sigmas[1], 2)
        amp2 = round(amps[1], 2)

        val = round(norm, 4)

        border = '--------------\n'
        total_number = '\n количество углов ' + str(val)
        images_number = '\n количество снимков ' + str(images_amount)
        text_angle = '\n шаг угла ' + str(step) + ' градусов'

        moda1 = '\n mu1 = ' + str(mu1) + ' sigma1 = ' + str(sigma1) + ' amp1 = ' + str(amp1)
        moda2 = '\n mu2 = ' + str(mu2) + ' sigma2 = ' + str(sigma2) + ' amp2 = ' + str(amp2)

        legend = border + name + ' ' + itype + total_number + images_number + text_angle + moda1 + moda2

        return legend

    @classmethod
    def angles_approx_save(self, images_path, save_path, types_dict, step, max_images_num_per_class=None, no_cache=False):
        """
        :param save_path:
        :param images: ndarray uint8 [[image1_class1,image2_class1,..],[image1_class2,image2_class2,..]..]
        :param paths:
        :param types_dict: list str [class_type1,class_type2,..]
        :param step: scalar int [0,N]
        :param max_images_num_per_class:
        """
        #
        # вычисление и сохранение распределения углов для всех фотографий одного образца
        #

        json_data = []

        dataset = SEMDataset(images_path, no_cache=no_cache, max_images_num_per_class=max_images_num_per_class)


        for i in tqdm(range(dataset.images_paths.shape[0])):
            all_angles = []
            # all_unique_angels=dict()

            for j in tqdm(range(dataset.images_paths.shape[1])):
                image, path = dataset.__getitem__(i,j)
                ang=grainMark.get_angles(image)

                # all_unique_angels[j]=ang
                all_angles.extend(ang)

            x, y = grainStats.stats_preprocess(all_angles, step)

            (x_gauss, y_gauss), mus, sigmas, amps = grainApprox.bimodal_gauss_approx(x, y)
            name = path.split('/')[-2]

            text = grainGenerate.angles_legend(dataset.images_paths.shape[1], types_dict[name], types_dict[name], step, mus, sigmas,amps, len(all_angles) )

            path='/'.join(path.split('/')[:-1])

            json_data.append({'path': path,
                              'name': name,
                              'type': types_dict[name],
                              'legend': text,
                              'density_curve_scatter': [x,y],
                              'gauss_approx_plot': [x_gauss, y_gauss],
                              'gauss_approx_data': {'mus': mus, 'sigmas':sigmas, 'amps':amps},
                              # 'angles_series': all_unique_angels,
                              })

        with open(f'{save_path}_step_{step}_angles.json', 'w', encoding='utf-8') as outfile:
            json.dump(json_data, outfile, cls=grainGenerate.NumpyEncoder, ensure_ascii=False)

    @classmethod
    def beams_legend(cls, images_amount, name, itype, norm, k, angle, b, score, dist_step, dist_mean):
        """
        :param name: str
        :param itype: str
        :param norm: int
        :param k: float
        :param angle: float
        :param b: float
        :param score: float
        :param dist_step: int
        :param dist_mean: float
        :return: str
        """
        #
        # создание легенды для распределения длин полуосей
        #
        border = '--------------'
        tp = '\n ' + name + ' тип ' + itype
        num = '\n регионы Co ' + str(norm) + ' шт'
        lin_k = '\n k наклона ' + str(round((k), 3)) + ' сдвиг b ' + str(round(b, 3))
        lin_k_angle = '\n угол наклона $' + str(round(angle, 3)) + '^{\circ}$'
        images_number_t = '\n количество снимков ' + str(images_amount)
        acc = '\n точность ' + str(round(score, 2))
        text_step = '\n шаг длины ' + str(dist_step) + '$ мкм$'
        mean_text = '\n средняя длина ' + str(round(dist_mean, 2))
        legend = border + tp + lin_k + lin_k_angle + images_number_t +acc + num + text_step + mean_text

        return legend

    @staticmethod
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float64) or isinstance(obj, np.float32):
                return float(obj)
            if isinstance(obj, np.int64) or isinstance(obj, np.int32): 
                return int(obj)  
            if isinstance(obj, np.ndarray): 
                return list(obj)  
            return json.JSONEncoder.default(self, obj)

    @classmethod
    def diametr_approx_save(cls, images_path, save_path, types_dict, step, pixel, start=2, end=-3,
                            debug=False, max_images_num_per_class=None, no_cache=False):
        #
        # вычисление и сохранение распределения длин а- и б- полуосей и угла поворота эллипса для разных образцов
        #

        json_data = []

        dataset = SEMDataset(images_path, no_cache=no_cache, max_images_num_per_class=max_images_num_per_class)

        for i in tqdm(range(dataset.images_paths.shape[0])):

            all_a_beams = []
            all_b_beams = []
            # all_angles = []
            # all_contours = []

            for j in tqdm(range(dataset.images_paths.shape[1])):
                image, path = dataset.__getitem__(i,j)
                b_beams, a_beams, angles, cetroids, contours = grainMark.get_mvee_params(image, 0.2, debug=debug)

                all_a_beams.extend(a_beams)
                all_b_beams.extend(b_beams)
                # all_angles.extend(angles)
                # all_contours.extend((contours))

            x1, y1 = grainStats.stats_preprocess(all_a_beams, step)
            x2, y2 = grainStats.stats_preprocess(all_b_beams, step)

            angles_set, angles_dens_curve = grainStats.stats_preprocess(np.rad2deg(angles).astype('int32'), step=step)

            x1 = x1.reshape(-1, 1) * pixel
            x2 = x2.reshape(-1, 1) * pixel
            # x_angles = np.array([angles_set]).reshape(-1, 1)

            y1 = np.log(y1).reshape(-1, 1)
            y2 = np.log(y2).reshape(-1, 1)
            # y_angles = angles_dens_curve.reshape(-1, 1)

            x1 = x1[start:end]
            x2 = x2[start:end]
            # x_angles = x_angles[start:end]

            y1 = y1[start:end]
            y2 = y2[start:end]
            # y_angles = y_angles[start:end]

            (x_pred1, y_pred1), k1, b1, angle1, score1 = grainApprox.lin_regr_approx(x1, y1)
            (x_pred2, y_pred2), k2, b2, angle2, score2 = grainApprox.lin_regr_approx(x2, y2)

            dist_step = pixel * step

            name = path.split('/')[-2]

            # dist1_set_mean = np.sum(dist1_set*dens1_curve) * pixel
            # dist2_set_mean = np.sum(dist2_set*dens2_curve) * pixel

            dist1_set_mean = 0
            dist2_set_mean = 0

            legend1 = grainGenerate.beams_legend(0, name, types_dict[name], len(all_a_beams), k1, angle1, b1, score1, dist_step, dist1_set_mean)
            legend2 = grainGenerate.beams_legend(0, name, types_dict[name], len(all_b_beams), k2, angle2, b2, score2, dist_step, dist2_set_mean)

            path='/'.join(path.split('/')[:-1])

            json_data.append({'path': path,
                              'name': name,
                              'type': types_dict[name],
                              'legend': [{'a_beams': legend1, 'b_beams': legend2}],
                              'density_curve_scatter': [
                                  {'a_beams': (x1.flatten(), y1.flatten()),
                                   'b_beams': (x2.flatten(), y2.flatten()),
                                   # 'angles': (x_angles.flatten(), y_angles.flatten())
                                   }
                              ],
                              'linear_approx_plot': [{'a_beams': (x_pred1.flatten(), y_pred1.flatten()),
                                                      'b_beams': (x_pred2.flatten(), y_pred2.flatten())}],
                              'linear_approx_data': [{'a_beams': {'k': k1, 'b': b1, 'angle': angle1, 'score': score1},
                                                      'b_beams': {'k': k2, 'b': b2, 'angle': angle2, 'score': score2}}],
                              # 'beams_length_series': [{'a_beams': all_a_beams, 'b_beams': all_b_beams}],
                              # 'angles_series': all_angles,
                              # 'contours_series': all_contours,
                              'pixel2meter': pixel,
                              })

        with open(f'{save_path}_step_{step}_beams.json', 'w', encoding='utf-8') as outfile:
            json.dump(json_data, outfile, cls=cls.NumpyEncoder, ensure_ascii=False)



class GrainLogs():

    @classmethod
    def printProgressBar(cls, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r",
                         eta=None):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix} ETA:{eta} s', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()


def paths_queues():

    workers = 23
    
    def do_job(tasks_to_accomplish,result_queue,  tqdm_queue):
        while not tasks_to_accomplish.empty():
            entry_node, exit_node = tasks_to_accomplish.get()
            paths = list(nx.all_simple_paths(g, source=entry_node, target=exit_node))
            tqdm_queue.put(1)
            result_queue.put([{'paths':paths,
                              'entry_node':entry_node,
                              'exit_node':exit_node,}])
    
    cart_list=[entry_nodes, exit_nodes]
    cart_list=[element for element in itertools.product(*cart_list)]
    
    number_of_tasks = len(cart_list)
    
    tasks_to_accomplish = Queue()
    result_queue = Queue()
    tqdm_queue = Queue()
    processes = []
    
    for i in range(number_of_tasks):
        tasks_to_accomplish.put(cart_list[i])
    
    for w in range(workers):
        p = Process(target=do_job, args=(tasks_to_accomplish, result_queue, tqdm_queue))
        processes.append(p)
        p.start()
        
    with tqdm(total=number_of_tasks) as pbar:
        completed = 0
        while completed < number_of_tasks:
            tqdm_queue.get()
            pbar.update(1)
            completed += 1
    
    results = []
    while not result_queue.empty():
        results.extend(result_queue.get())
    
    for p in processes:
        p.join()
