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
    sphere { m*<0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 1 }        
    sphere {  m*<0.2769811153240989,-1.429098012755983e-18,4.107874587336951>, 1 }
    sphere {  m*<8.484565988246263,5.304508724251513e-18,-1.9076605428303648>, 1 }
    sphere {  m*<-4.506308737888632,8.164965809277259,-2.173328680646364>, 1}
    sphere { m*<-4.506308737888632,-8.164965809277259,-2.1733286806463674>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2769811153240989,-1.429098012755983e-18,4.107874587336951>, <0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 0.5 }
    cylinder { m*<8.484565988246263,5.304508724251513e-18,-1.9076605428303648>, <0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 0.5}
    cylinder { m*<-4.506308737888632,8.164965809277259,-2.173328680646364>, <0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 0.5 }
    cylinder {  m*<-4.506308737888632,-8.164965809277259,-2.1733286806463674>, <0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 0.5}

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
    sphere { m*<0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 1 }        
    sphere {  m*<0.2769811153240989,-1.429098012755983e-18,4.107874587336951>, 1 }
    sphere {  m*<8.484565988246263,5.304508724251513e-18,-1.9076605428303648>, 1 }
    sphere {  m*<-4.506308737888632,8.164965809277259,-2.173328680646364>, 1}
    sphere { m*<-4.506308737888632,-8.164965809277259,-2.1733286806463674>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2769811153240989,-1.429098012755983e-18,4.107874587336951>, <0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 0.5 }
    cylinder { m*<8.484565988246263,5.304508724251513e-18,-1.9076605428303648>, <0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 0.5}
    cylinder { m*<-4.506308737888632,8.164965809277259,-2.173328680646364>, <0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 0.5 }
    cylinder {  m*<-4.506308737888632,-8.164965809277259,-2.1733286806463674>, <0.24404723219576152,-2.5129031200910982e-18,1.1080543318808005>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    