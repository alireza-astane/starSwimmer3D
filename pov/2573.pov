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
    sphere { m*<0.8318472807833446,0.7330039969678855,0.3577112925440933>, 1 }        
    sphere {  m*<1.0750825516263016,0.7968254642021081,3.3471507023808167>, 1 }
    sphere {  m*<3.568329740688838,0.7968254642021079,-0.8701315061097994>, 1 }
    sphere {  m*<-2.29411604029678,5.486593112215553,-1.4905521941984168>, 1}
    sphere { m*<-3.8751621403481766,-7.639155705907625,-2.4247067059732617>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0750825516263016,0.7968254642021081,3.3471507023808167>, <0.8318472807833446,0.7330039969678855,0.3577112925440933>, 0.5 }
    cylinder { m*<3.568329740688838,0.7968254642021079,-0.8701315061097994>, <0.8318472807833446,0.7330039969678855,0.3577112925440933>, 0.5}
    cylinder { m*<-2.29411604029678,5.486593112215553,-1.4905521941984168>, <0.8318472807833446,0.7330039969678855,0.3577112925440933>, 0.5 }
    cylinder {  m*<-3.8751621403481766,-7.639155705907625,-2.4247067059732617>, <0.8318472807833446,0.7330039969678855,0.3577112925440933>, 0.5}

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
    sphere { m*<0.8318472807833446,0.7330039969678855,0.3577112925440933>, 1 }        
    sphere {  m*<1.0750825516263016,0.7968254642021081,3.3471507023808167>, 1 }
    sphere {  m*<3.568329740688838,0.7968254642021079,-0.8701315061097994>, 1 }
    sphere {  m*<-2.29411604029678,5.486593112215553,-1.4905521941984168>, 1}
    sphere { m*<-3.8751621403481766,-7.639155705907625,-2.4247067059732617>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0750825516263016,0.7968254642021081,3.3471507023808167>, <0.8318472807833446,0.7330039969678855,0.3577112925440933>, 0.5 }
    cylinder { m*<3.568329740688838,0.7968254642021079,-0.8701315061097994>, <0.8318472807833446,0.7330039969678855,0.3577112925440933>, 0.5}
    cylinder { m*<-2.29411604029678,5.486593112215553,-1.4905521941984168>, <0.8318472807833446,0.7330039969678855,0.3577112925440933>, 0.5 }
    cylinder {  m*<-3.8751621403481766,-7.639155705907625,-2.4247067059732617>, <0.8318472807833446,0.7330039969678855,0.3577112925440933>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    