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
    sphere { m*<-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 1 }        
    sphere {  m*<0.35557056849641633,-0.15929553669114907,9.101858712176961>, 1 }
    sphere {  m*<7.710922006496392,-0.2482158126855054,-5.477634577868377>, 1 }
    sphere {  m*<-5.802117887539829,4.829641293420382,-3.18080158350493>, 1}
    sphere { m*<-2.3421724810562794,-3.5901239724040246,-1.3837312270744904>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.35557056849641633,-0.15929553669114907,9.101858712176961>, <-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 0.5 }
    cylinder { m*<7.710922006496392,-0.2482158126855054,-5.477634577868377>, <-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 0.5}
    cylinder { m*<-5.802117887539829,4.829641293420382,-3.18080158350493>, <-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 0.5 }
    cylinder {  m*<-2.3421724810562794,-3.5901239724040246,-1.3837312270744904>, <-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 0.5}

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
    sphere { m*<-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 1 }        
    sphere {  m*<0.35557056849641633,-0.15929553669114907,9.101858712176961>, 1 }
    sphere {  m*<7.710922006496392,-0.2482158126855054,-5.477634577868377>, 1 }
    sphere {  m*<-5.802117887539829,4.829641293420382,-3.18080158350493>, 1}
    sphere { m*<-2.3421724810562794,-3.5901239724040246,-1.3837312270744904>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.35557056849641633,-0.15929553669114907,9.101858712176961>, <-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 0.5 }
    cylinder { m*<7.710922006496392,-0.2482158126855054,-5.477634577868377>, <-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 0.5}
    cylinder { m*<-5.802117887539829,4.829641293420382,-3.18080158350493>, <-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 0.5 }
    cylinder {  m*<-2.3421724810562794,-3.5901239724040246,-1.3837312270744904>, <-1.0810206775158697,-0.9399670353350649,-0.7636477299044366>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    