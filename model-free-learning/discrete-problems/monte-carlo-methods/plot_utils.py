import numpy as np
import matplotlib.pyplot as plt

def plot_blackjack_values(V):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


def plot_policy(policy):

    def get_Z(x, y, usable_ace):
        if (x, y, usable_ace) in policy:
            return policy[x, y, usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y, usable_ace) for x in x_range] for y in y_range])
        
        surf = ax.imshow(
            Z.T,
            cmap=plt.get_cmap('Pastel2', 2),
            vmin=0,
            vmax=1,
            extent=[0.5, 10.5, 10.5, 21.5],
            origin='lower'
        )
        
        ax.set_xticks(np.arange(1, 11))
        ax.set_yticks(np.arange(11, 22))
        ax.set_xticklabels(np.arange(1, 11,))
        ax.set_yticklabels(np.arange(11, 22))

        ax.set_xlabel('Dealer\'s Showing Card')
        ax.set_ylabel('Player\'s Current Sum')

        ax.grid(color='w', linestyle='-', linewidth=1)

        # Add color bar
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])

    # Create figure
    fig = plt.figure(figsize=(15, 15))
    
    # Usable Ace
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    
    # No Usable Ace
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    
    plt.show()