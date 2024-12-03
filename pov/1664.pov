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
    sphere { m*<0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 1 }        
    sphere {  m*<1.0400786938098618,1.4184040310585667e-18,3.8297715732438844>, 1 }
    sphere {  m*<5.8368726537104445,5.192359158422735e-18,-1.195318784872162>, 1 }
    sphere {  m*<-3.981177362736068,8.164965809277259,-2.2628923395178058>, 1}
    sphere { m*<-3.981177362736068,-8.164965809277259,-2.2628923395178093>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0400786938098618,1.4184040310585667e-18,3.8297715732438844>, <0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 0.5 }
    cylinder { m*<5.8368726537104445,5.192359158422735e-18,-1.195318784872162>, <0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 0.5}
    cylinder { m*<-3.981177362736068,8.164965809277259,-2.2628923395178058>, <0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 0.5 }
    cylinder {  m*<-3.981177362736068,-8.164965809277259,-2.2628923395178093>, <0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 0.5}

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
    sphere { m*<0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 1 }        
    sphere {  m*<1.0400786938098618,1.4184040310585667e-18,3.8297715732438844>, 1 }
    sphere {  m*<5.8368726537104445,5.192359158422735e-18,-1.195318784872162>, 1 }
    sphere {  m*<-3.981177362736068,8.164965809277259,-2.2628923395178058>, 1}
    sphere { m*<-3.981177362736068,-8.164965809277259,-2.2628923395178093>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0400786938098618,1.4184040310585667e-18,3.8297715732438844>, <0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 0.5 }
    cylinder { m*<5.8368726537104445,5.192359158422735e-18,-1.195318784872162>, <0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 0.5}
    cylinder { m*<-3.981177362736068,8.164965809277259,-2.2628923395178058>, <0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 0.5 }
    cylinder {  m*<-3.981177362736068,-8.164965809277259,-2.2628923395178093>, <0.8918340032705206,-2.0037357651530283e-18,0.8334309556670125>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    