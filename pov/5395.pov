#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 1 }        
    sphere {  m*<0.3219821202724091,0.2862328411136116,8.466305600617106>, 1 }
    sphere {  m*<4.43641626493044,0.030070551938275653,-3.9758700359588044>, 1 }
    sphere {  m*<-2.4032902532271536,2.1718588658514837,-2.3973770795273714>, 1}
    sphere { m*<-2.1355030321893214,-2.7158330765524137,-2.207830794364801>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3219821202724091,0.2862328411136116,8.466305600617106>, <-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 0.5 }
    cylinder { m*<4.43641626493044,0.030070551938275653,-3.9758700359588044>, <-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 0.5}
    cylinder { m*<-2.4032902532271536,2.1718588658514837,-2.3973770795273714>, <-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 0.5 }
    cylinder {  m*<-2.1355030321893214,-2.7158330765524137,-2.207830794364801>, <-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 1 }        
    sphere {  m*<0.3219821202724091,0.2862328411136116,8.466305600617106>, 1 }
    sphere {  m*<4.43641626493044,0.030070551938275653,-3.9758700359588044>, 1 }
    sphere {  m*<-2.4032902532271536,2.1718588658514837,-2.3973770795273714>, 1}
    sphere { m*<-2.1355030321893214,-2.7158330765524137,-2.207830794364801>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3219821202724091,0.2862328411136116,8.466305600617106>, <-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 0.5 }
    cylinder { m*<4.43641626493044,0.030070551938275653,-3.9758700359588044>, <-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 0.5}
    cylinder { m*<-2.4032902532271536,2.1718588658514837,-2.3973770795273714>, <-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 0.5 }
    cylinder {  m*<-2.1355030321893214,-2.7158330765524137,-2.207830794364801>, <-0.7576653945672935,-0.15692280743019726,-1.4653426964818883>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    