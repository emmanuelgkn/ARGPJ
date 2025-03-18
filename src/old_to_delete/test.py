import pygame
import gymnasium
import gym_MPR

#fichier pour tester le jeu au test
def main():
    pygame.init()
    env = gymnasium.make("MatPodRacer-v0",render_enabled=True)
    env.reset()
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        action = None
        if keys[pygame.K_UP]:
            action=0
        elif keys[pygame.K_DOWN]:
            action=1 
        elif keys[pygame.K_LEFT]:
            action=2 
        elif keys[pygame.K_RIGHT]:
            action=3 
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            action=4  
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            action=5  
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
            action=6  
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
            action=7 
        
        if action is not None:
            env.step(action)
        env.render()
        
        clock.tick(6)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()