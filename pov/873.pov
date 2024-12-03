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
    sphere { m*<-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 1 }        
    sphere {  m*<-2.2606734061461752e-18,-4.2247099960783514e-18,4.944128968387204>, 1 }
    sphere {  m*<9.428090415820634,-2.1509887021707168e-20,-2.2712043649461635>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.2712043649461635>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.2712043649461635>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.2606734061461752e-18,-4.2247099960783514e-18,4.944128968387204>, <-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 0.5 }
    cylinder { m*<9.428090415820634,-2.1509887021707168e-20,-2.2712043649461635>, <-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.2712043649461635>, <-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.2712043649461635>, <-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 0.5}

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
    sphere { m*<-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 1 }        
    sphere {  m*<-2.2606734061461752e-18,-4.2247099960783514e-18,4.944128968387204>, 1 }
    sphere {  m*<9.428090415820634,-2.1509887021707168e-20,-2.2712043649461635>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.2712043649461635>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.2712043649461635>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.2606734061461752e-18,-4.2247099960783514e-18,4.944128968387204>, <-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 0.5 }
    cylinder { m*<9.428090415820634,-2.1509887021707168e-20,-2.2712043649461635>, <-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.2712043649461635>, <-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.2712043649461635>, <-2.206550075436271e-18,-5.220533610564296e-18,1.0621289683871702>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    