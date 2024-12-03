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
    sphere { m*<-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 1 }        
    sphere {  m*<-0.06619872247099257,0.19274276633189374,8.886915657227352>, 1 }
    sphere {  m*<7.289152715528982,0.10382249033753638,-5.6925776328180095>, 1 }
    sphere {  m*<-3.642485201909299,2.5700502409704384,-2.077411747025267>, 1}
    sphere { m*<-2.909272436408144,-2.869156698503046,-1.6742512646167742>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.06619872247099257,0.19274276633189374,8.886915657227352>, <-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 0.5 }
    cylinder { m*<7.289152715528982,0.10382249033753638,-5.6925776328180095>, <-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 0.5}
    cylinder { m*<-3.642485201909299,2.5700502409704384,-2.077411747025267>, <-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 0.5 }
    cylinder {  m*<-2.909272436408144,-2.869156698503046,-1.6742512646167742>, <-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 0.5}

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
    sphere { m*<-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 1 }        
    sphere {  m*<-0.06619872247099257,0.19274276633189374,8.886915657227352>, 1 }
    sphere {  m*<7.289152715528982,0.10382249033753638,-5.6925776328180095>, 1 }
    sphere {  m*<-3.642485201909299,2.5700502409704384,-2.077411747025267>, 1}
    sphere { m*<-2.909272436408144,-2.869156698503046,-1.6742512646167742>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.06619872247099257,0.19274276633189374,8.886915657227352>, <-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 0.5 }
    cylinder { m*<7.289152715528982,0.10382249033753638,-5.6925776328180095>, <-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 0.5}
    cylinder { m*<-3.642485201909299,2.5700502409704384,-2.077411747025267>, <-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 0.5 }
    cylinder {  m*<-2.909272436408144,-2.869156698503046,-1.6742512646167742>, <-1.5289980758774249,-0.2943033740332095,-0.9936076739452216>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    