void add_object(int map[MAP_H][MAP_W], int N, const char* s) {
    

    pos_t locations = malloc(N * sizeof(pos_t));

   
    for (int i = 0; i < N; ++i) {
        do {
            locations[i].r = random(MAP_H) % MAP_H; 
            locations[i].c = random(MAP_W) % MAP_W;
        } while (is_in_array(locations, i, locations[i].r, locations[i].c));

        
        map[locations[i].r][locations[i].c] = 1; 
    }

   
}


bool is_in_array(pos_t *array, int size, int r, int c) {
    for (int i = 0; i < size; ++i) {
        if (array[i].r == r && array[i].c == c) {
            return true;
        }
    }
    return false;
}


void initialize_all(int map[MAP_H][MAP_W]) {
    for (int r = 0; r < MAP_H; ++r) {
        for (int c = 0; c < MAP_W; ++c) {
            map[r][c] = 0;
        }
    }

    
    add_object(map, N_FIRES, "Fire");
}