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
    sphere { m*<-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 1 }        
    sphere {  m*<1.0975916922745035,0.7690693751590516,9.450808341106228>, 1 }
    sphere {  m*<8.46537889059729,0.48397712436678897,-5.119869087967694>, 1 }
    sphere {  m*<-6.430584303091693,7.007058497987425,-3.629062184786089>, 1}
    sphere { m*<-4.3842737325240835,-9.06863235804188,-2.2798665844445347>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0975916922745035,0.7690693751590516,9.450808341106228>, <-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 0.5 }
    cylinder { m*<8.46537889059729,0.48397712436678897,-5.119869087967694>, <-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 0.5}
    cylinder { m*<-6.430584303091693,7.007058497987425,-3.629062184786089>, <-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 0.5 }
    cylinder {  m*<-4.3842737325240835,-9.06863235804188,-2.2798665844445347>, <-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 0.5}

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
    sphere { m*<-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 1 }        
    sphere {  m*<1.0975916922745035,0.7690693751590516,9.450808341106228>, 1 }
    sphere {  m*<8.46537889059729,0.48397712436678897,-5.119869087967694>, 1 }
    sphere {  m*<-6.430584303091693,7.007058497987425,-3.629062184786089>, 1}
    sphere { m*<-4.3842737325240835,-9.06863235804188,-2.2798665844445347>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0975916922745035,0.7690693751590516,9.450808341106228>, <-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 0.5 }
    cylinder { m*<8.46537889059729,0.48397712436678897,-5.119869087967694>, <-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 0.5}
    cylinder { m*<-6.430584303091693,7.007058497987425,-3.629062184786089>, <-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 0.5 }
    cylinder {  m*<-4.3842737325240835,-9.06863235804188,-2.2798665844445347>, <-0.32157580192565727,-0.22086953872086568,-0.39848175592891233>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    