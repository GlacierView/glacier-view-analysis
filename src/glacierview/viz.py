import matplotlib.pyplot as plt


def plot_examples(rasters,masks,glac_ids, n = None,bands = 6, where = 0):
    for i in range(n):
        print(i, " ", glac_ids[where + i])

        _, axs = plt.subplots(ncols = bands + 1,figsize=(20,20))
        for j in range(bands):
            axs[j].imshow(rasters[where + i][j])
        axs[bands].imshow(masks[where + i][j])
        plt.show()

def view_training_images(X_train, n = None, where = 0):
    """
    X_train: (np array) example,dim1,dim2,channel
    Channel: (int list) which channel we want to see
    n: (int) number of images to view
    where: (int) where to view those n images

    """
    if n is None:
         n = X_train.shape[0]

    bands = X_train.shape[3]
    for i in range(n):
        _, axs = plt.subplots(ncols=bands,figsize=(20,20))
        for j in range(bands):
            axs[j].imshow(X_train[where + i,:,:,j])
        plt.show()


def view_ts_images(X_eval_ts, dates,  n = None, where = 0):
    """
    X_eval_ts: (np array) example,dim1,dim2,channel
    dates: (datetime list) dates of each image
    Channel: (int list) which channel we want to see
    n: (int) number of images to view
    where: (int) where to view those n images

    """
    if n is None:
         n = len(dates)-where

    bands = X_eval_ts.shape[3]
    print(bands)
    if bands == 1:
        for i in range(n):
            print(dates[where+i], where + i)
            plt.imshow(X_eval_ts[where + i,:,:,0])
            plt.show()
    else:
        for i in range(n):
            _, axs = plt.subplots(ncols = bands,figsize=(20,20))
            print(dates[where + i], where + i)
            for j in range(bands):
                axs[j].imshow(X_eval_ts[where + i,:,:,j])
            plt.show()
