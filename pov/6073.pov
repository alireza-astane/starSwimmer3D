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
    sphere { m*<-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 1 }        
    sphere {  m*<-0.08328180601616686,0.2074541619986749,8.878206969997509>, 1 }
    sphere {  m*<7.272069631983807,0.11853388600431747,-5.7012863200478545>, 1 }
    sphere {  m*<-3.538449784039786,2.451204367655916,-2.024197358288911>, 1}
    sphere { m*<-2.933932909244292,-2.833708713218467,-1.6869096295518036>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08328180601616686,0.2074541619986749,8.878206969997509>, <-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 0.5 }
    cylinder { m*<7.272069631983807,0.11853388600431747,-5.7012863200478545>, <-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 0.5}
    cylinder { m*<-3.538449784039786,2.451204367655916,-2.024197358288911>, <-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 0.5 }
    cylinder {  m*<-2.933932909244292,-2.833708713218467,-1.6869096295518036>, <-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 0.5}

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
    sphere { m*<-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 1 }        
    sphere {  m*<-0.08328180601616686,0.2074541619986749,8.878206969997509>, 1 }
    sphere {  m*<7.272069631983807,0.11853388600431747,-5.7012863200478545>, 1 }
    sphere {  m*<-3.538449784039786,2.451204367655916,-2.024197358288911>, 1}
    sphere { m*<-2.933932909244292,-2.833708713218467,-1.6869096295518036>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08328180601616686,0.2074541619986749,8.878206969997509>, <-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 0.5 }
    cylinder { m*<7.272069631983807,0.11853388600431747,-5.7012863200478545>, <-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 0.5}
    cylinder { m*<-3.538449784039786,2.451204367655916,-2.024197358288911>, <-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 0.5 }
    cylinder {  m*<-2.933932909244292,-2.833708713218467,-1.6869096295518036>, <-1.547143551852031,-0.26323826574247455,-1.0029522979406138>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    