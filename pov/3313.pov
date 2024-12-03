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
    sphere { m*<0.2791838114215736,0.7356353713852499,0.03370688138285513>, 1 }        
    sphere {  m*<0.5199189161632652,0.8643454495655756,3.021261652503405>, 1 }
    sphere {  m*<3.0138922054278297,0.8376693467716245,-1.1955026440683278>, 1 }
    sphere {  m*<-1.3424315484713172,3.064109315803853,-0.9402388840331141>, 1}
    sphere { m*<-3.241671901694147,-5.920037568708506,-2.0062547002628044>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5199189161632652,0.8643454495655756,3.021261652503405>, <0.2791838114215736,0.7356353713852499,0.03370688138285513>, 0.5 }
    cylinder { m*<3.0138922054278297,0.8376693467716245,-1.1955026440683278>, <0.2791838114215736,0.7356353713852499,0.03370688138285513>, 0.5}
    cylinder { m*<-1.3424315484713172,3.064109315803853,-0.9402388840331141>, <0.2791838114215736,0.7356353713852499,0.03370688138285513>, 0.5 }
    cylinder {  m*<-3.241671901694147,-5.920037568708506,-2.0062547002628044>, <0.2791838114215736,0.7356353713852499,0.03370688138285513>, 0.5}

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
    sphere { m*<0.2791838114215736,0.7356353713852499,0.03370688138285513>, 1 }        
    sphere {  m*<0.5199189161632652,0.8643454495655756,3.021261652503405>, 1 }
    sphere {  m*<3.0138922054278297,0.8376693467716245,-1.1955026440683278>, 1 }
    sphere {  m*<-1.3424315484713172,3.064109315803853,-0.9402388840331141>, 1}
    sphere { m*<-3.241671901694147,-5.920037568708506,-2.0062547002628044>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5199189161632652,0.8643454495655756,3.021261652503405>, <0.2791838114215736,0.7356353713852499,0.03370688138285513>, 0.5 }
    cylinder { m*<3.0138922054278297,0.8376693467716245,-1.1955026440683278>, <0.2791838114215736,0.7356353713852499,0.03370688138285513>, 0.5}
    cylinder { m*<-1.3424315484713172,3.064109315803853,-0.9402388840331141>, <0.2791838114215736,0.7356353713852499,0.03370688138285513>, 0.5 }
    cylinder {  m*<-3.241671901694147,-5.920037568708506,-2.0062547002628044>, <0.2791838114215736,0.7356353713852499,0.03370688138285513>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    