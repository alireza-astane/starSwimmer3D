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
    sphere { m*<0.7786199951781942,0.8115403567751839,0.3262401203700303>, 1 }        
    sphere {  m*<1.0215708807531945,0.8837686001002982,3.3155110460961126>, 1 }
    sphere {  m*<3.514818069815729,0.883768600100298,-0.9017711623945013>, 1 }
    sphere {  m*<-2.116886868794302,5.170571598655366,-1.385761339765381>, 1}
    sphere { m*<-3.893329827502821,-7.587085712310997,-2.4354496056614536>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0215708807531945,0.8837686001002982,3.3155110460961126>, <0.7786199951781942,0.8115403567751839,0.3262401203700303>, 0.5 }
    cylinder { m*<3.514818069815729,0.883768600100298,-0.9017711623945013>, <0.7786199951781942,0.8115403567751839,0.3262401203700303>, 0.5}
    cylinder { m*<-2.116886868794302,5.170571598655366,-1.385761339765381>, <0.7786199951781942,0.8115403567751839,0.3262401203700303>, 0.5 }
    cylinder {  m*<-3.893329827502821,-7.587085712310997,-2.4354496056614536>, <0.7786199951781942,0.8115403567751839,0.3262401203700303>, 0.5}

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
    sphere { m*<0.7786199951781942,0.8115403567751839,0.3262401203700303>, 1 }        
    sphere {  m*<1.0215708807531945,0.8837686001002982,3.3155110460961126>, 1 }
    sphere {  m*<3.514818069815729,0.883768600100298,-0.9017711623945013>, 1 }
    sphere {  m*<-2.116886868794302,5.170571598655366,-1.385761339765381>, 1}
    sphere { m*<-3.893329827502821,-7.587085712310997,-2.4354496056614536>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0215708807531945,0.8837686001002982,3.3155110460961126>, <0.7786199951781942,0.8115403567751839,0.3262401203700303>, 0.5 }
    cylinder { m*<3.514818069815729,0.883768600100298,-0.9017711623945013>, <0.7786199951781942,0.8115403567751839,0.3262401203700303>, 0.5}
    cylinder { m*<-2.116886868794302,5.170571598655366,-1.385761339765381>, <0.7786199951781942,0.8115403567751839,0.3262401203700303>, 0.5 }
    cylinder {  m*<-3.893329827502821,-7.587085712310997,-2.4354496056614536>, <0.7786199951781942,0.8115403567751839,0.3262401203700303>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    