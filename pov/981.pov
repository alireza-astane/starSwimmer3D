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
    sphere { m*<-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 1 }        
    sphere {  m*<-4.5012985567959044e-18,-5.491614910548607e-18,4.3031659417758314>, 1 }
    sphere {  m*<9.428090415820634,8.193873774493712e-21,-2.156167391557549>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.156167391557549>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.156167391557549>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.5012985567959044e-18,-5.491614910548607e-18,4.3031659417758314>, <-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 0.5 }
    cylinder { m*<9.428090415820634,8.193873774493712e-21,-2.156167391557549>, <-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.156167391557549>, <-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.156167391557549>, <-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 0.5}

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
    sphere { m*<-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 1 }        
    sphere {  m*<-4.5012985567959044e-18,-5.491614910548607e-18,4.3031659417758314>, 1 }
    sphere {  m*<9.428090415820634,8.193873774493712e-21,-2.156167391557549>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.156167391557549>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.156167391557549>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.5012985567959044e-18,-5.491614910548607e-18,4.3031659417758314>, <-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 0.5 }
    cylinder { m*<9.428090415820634,8.193873774493712e-21,-2.156167391557549>, <-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.156167391557549>, <-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.156167391557549>, <-1.1631415844164378e-18,-5.57283036145084e-18,1.1771659417757843>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    