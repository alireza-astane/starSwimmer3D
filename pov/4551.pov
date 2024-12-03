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
    sphere { m*<-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 1 }        
    sphere {  m*<0.34157709901347905,0.18740789299931088,5.883316491303282>, 1 }
    sphere {  m*<2.5254835733707464,-0.005046790490898886,-2.181418350471715>, 1 }
    sphere {  m*<-1.830840180528401,2.2213931785413257,-1.9261545904365023>, 1}
    sphere { m*<-1.563052959490569,-2.6662987638625717,-1.7366083052739294>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.34157709901347905,0.18740789299931088,5.883316491303282>, <-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 0.5 }
    cylinder { m*<2.5254835733707464,-0.005046790490898886,-2.181418350471715>, <-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 0.5}
    cylinder { m*<-1.830840180528401,2.2213931785413257,-1.9261545904365023>, <-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 0.5 }
    cylinder {  m*<-1.563052959490569,-2.6662987638625717,-1.7366083052739294>, <-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 0.5}

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
    sphere { m*<-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 1 }        
    sphere {  m*<0.34157709901347905,0.18740789299931088,5.883316491303282>, 1 }
    sphere {  m*<2.5254835733707464,-0.005046790490898886,-2.181418350471715>, 1 }
    sphere {  m*<-1.830840180528401,2.2213931785413257,-1.9261545904365023>, 1}
    sphere { m*<-1.563052959490569,-2.6662987638625717,-1.7366083052739294>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.34157709901347905,0.18740789299931088,5.883316491303282>, <-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 0.5 }
    cylinder { m*<2.5254835733707464,-0.005046790490898886,-2.181418350471715>, <-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 0.5}
    cylinder { m*<-1.830840180528401,2.2213931785413257,-1.9261545904365023>, <-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 0.5 }
    cylinder {  m*<-1.563052959490569,-2.6662987638625717,-1.7366083052739294>, <-0.20922482063551093,-0.10708076587727311,-0.9522088250205347>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    