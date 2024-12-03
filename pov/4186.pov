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
    sphere { m*<-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 1 }        
    sphere {  m*<0.17828976391786377,0.10010560408484392,3.856898875343589>, 1 }
    sphere {  m*<2.567222302480138,0.017269003844877687,-1.6634351530270768>, 1 }
    sphere {  m*<-1.7891014514190091,2.2437089728771022,-1.4081713929918633>, 1}
    sphere { m*<-1.5213142303811773,-2.643982969526795,-1.218625107829291>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17828976391786377,0.10010560408484392,3.856898875343589>, <-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 0.5 }
    cylinder { m*<2.567222302480138,0.017269003844877687,-1.6634351530270768>, <-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 0.5}
    cylinder { m*<-1.7891014514190091,2.2437089728771022,-1.4081713929918633>, <-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 0.5 }
    cylinder {  m*<-1.5213142303811773,-2.643982969526795,-1.218625107829291>, <-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 0.5}

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
    sphere { m*<-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 1 }        
    sphere {  m*<0.17828976391786377,0.10010560408484392,3.856898875343589>, 1 }
    sphere {  m*<2.567222302480138,0.017269003844877687,-1.6634351530270768>, 1 }
    sphere {  m*<-1.7891014514190091,2.2437089728771022,-1.4081713929918633>, 1}
    sphere { m*<-1.5213142303811773,-2.643982969526795,-1.218625107829291>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17828976391786377,0.10010560408484392,3.856898875343589>, <-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 0.5 }
    cylinder { m*<2.567222302480138,0.017269003844877687,-1.6634351530270768>, <-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 0.5}
    cylinder { m*<-1.7891014514190091,2.2437089728771022,-1.4081713929918633>, <-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 0.5 }
    cylinder {  m*<-1.5213142303811773,-2.643982969526795,-1.218625107829291>, <-0.16748609152611904,-0.08476497154149645,-0.4342256275758938>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    