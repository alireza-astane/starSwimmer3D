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
    sphere { m*<0.41848159269287666,0.9989578557372353,0.11441513680593081>, 1 }        
    sphere {  m*<0.6592166974345683,1.1276679339175608,3.101969907926482>, 1 }
    sphere {  m*<3.153189986699133,1.10099183112361,-1.1147943886452536>, 1 }
    sphere {  m*<-1.203133767200013,3.3274318001558356,-0.8595306286100394>, 1}
    sphere { m*<-3.7025712619819213,-6.791301730130012,-2.2732968810854004>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6592166974345683,1.1276679339175608,3.101969907926482>, <0.41848159269287666,0.9989578557372353,0.11441513680593081>, 0.5 }
    cylinder { m*<3.153189986699133,1.10099183112361,-1.1147943886452536>, <0.41848159269287666,0.9989578557372353,0.11441513680593081>, 0.5}
    cylinder { m*<-1.203133767200013,3.3274318001558356,-0.8595306286100394>, <0.41848159269287666,0.9989578557372353,0.11441513680593081>, 0.5 }
    cylinder {  m*<-3.7025712619819213,-6.791301730130012,-2.2732968810854004>, <0.41848159269287666,0.9989578557372353,0.11441513680593081>, 0.5}

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
    sphere { m*<0.41848159269287666,0.9989578557372353,0.11441513680593081>, 1 }        
    sphere {  m*<0.6592166974345683,1.1276679339175608,3.101969907926482>, 1 }
    sphere {  m*<3.153189986699133,1.10099183112361,-1.1147943886452536>, 1 }
    sphere {  m*<-1.203133767200013,3.3274318001558356,-0.8595306286100394>, 1}
    sphere { m*<-3.7025712619819213,-6.791301730130012,-2.2732968810854004>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6592166974345683,1.1276679339175608,3.101969907926482>, <0.41848159269287666,0.9989578557372353,0.11441513680593081>, 0.5 }
    cylinder { m*<3.153189986699133,1.10099183112361,-1.1147943886452536>, <0.41848159269287666,0.9989578557372353,0.11441513680593081>, 0.5}
    cylinder { m*<-1.203133767200013,3.3274318001558356,-0.8595306286100394>, <0.41848159269287666,0.9989578557372353,0.11441513680593081>, 0.5 }
    cylinder {  m*<-3.7025712619819213,-6.791301730130012,-2.2732968810854004>, <0.41848159269287666,0.9989578557372353,0.11441513680593081>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    