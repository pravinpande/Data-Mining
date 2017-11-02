import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_dataset(name):
    return np.loadtxt(name)



def euclidian(a, b):
    return np.linalg.norm(a-b)


def plot(dataset, history_centroids, belongs_to):
    colors = ['r', 'g']

    fig, ax = plt.subplots()

    for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    history_points = []
    for index, centroids in enumerate(history_centroids):
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))

                plt.pause(0.8)


def kmeans(k, epsilon=0, distance='euclidian'):
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian
    dataset = load_dataset('faithful.txt')
    # dataset = dataset[:, 0:dataset.shape[1] - 1]
    num_instances, num_features = dataset.shape
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    belongs_to = np.zeros((num_instances, 1))
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)

            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(dataset[instances_close], axis=0)
            # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes

        history_centroids.append(tmp_prototypes)

    # plot(dataset, history_centroids, belongs_to)

    return prototypes, history_centroids, belongs_to

###############


def execute():
    dataset = load_dataset('faithful.txt')
    centroids, history_centroids, belongs_to = kmeans(2)
    plot(dataset, history_centroids, belongs_to)

execute()



def pick_mu(k):
    dataset = load_dataset('faithful.txt')
    x_copy=dataset
    np.random.shuffle(x_copy)
    mu=x_copy[:k]
    return mu

def obj(k):
    L=[ ]
    mu=pick_mu(k)
    c=np.zeros(500).astype(int)
    c=c.astype(int)
    
    x = load_dataset('faithful.txt')
    dataset= np.vstack((x))
    for t in range(20):

        for i in range(dataset.shape[0]):
            dist=[]
            for k1 in range(k):
                dist.append(np.sqrt(np.sum((dataset[i]-mu[k1])**2)))
            c[i]=np.argmin(dist)+1

        for k1 in range(k):
            nk=0
            for i in range(dataset.shape[0]):
                if c[i]==k1+1:
                    nk=nk+1
            val =np.zeros((1,2))
            for i in range(dataset.shape[0]):
                if c[i]==k1+1:
                    val=np.add(val,dataset[i])

            mu[k1]=val/nk

        sum1=0
        for i in range(dataset.shape[0]):
            for k1 in range(k):
                if c[i]==k1+1:
                    sum1=sum1+(np.sum((dataset[i]-mu[k1])**2))
        L.append(sum1)
    return L,mu,c
            
x_axis = np.arange(1, 21)

plt.figure(figsize=(9,6))

l2,mu2,c2=obj(2)
plt.plot(x_axis,l2, label="k=2")

plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
plt.title("Objective Training Function L per iteration --->")
plt.xlabel("No of iterations --->")
plt.ylabel("Objective Training Function")
plt.show()
