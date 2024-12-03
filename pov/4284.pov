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
    sphere { m*<-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 1 }        
    sphere {  m*<0.2228841008065262,0.12394816162749331,4.410320546352619>, 1 }
    sphere {  m*<2.5567685454178672,0.011679856843626102,-1.793167673014278>, 1 }
    sphere {  m*<-1.79955520848128,2.2381198258758506,-1.5379039129790646>, 1}
    sphere { m*<-1.5317679874434482,-2.649572116528047,-1.3483576278164922>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2228841008065262,0.12394816162749331,4.410320546352619>, <-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 0.5 }
    cylinder { m*<2.5567685454178672,0.011679856843626102,-1.793167673014278>, <-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 0.5}
    cylinder { m*<-1.79955520848128,2.2381198258758506,-1.5379039129790646>, <-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 0.5 }
    cylinder {  m*<-1.5317679874434482,-2.649572116528047,-1.3483576278164922>, <-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 0.5}

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
    sphere { m*<-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 1 }        
    sphere {  m*<0.2228841008065262,0.12394816162749331,4.410320546352619>, 1 }
    sphere {  m*<2.5567685454178672,0.011679856843626102,-1.793167673014278>, 1 }
    sphere {  m*<-1.79955520848128,2.2381198258758506,-1.5379039129790646>, 1}
    sphere { m*<-1.5317679874434482,-2.649572116528047,-1.3483576278164922>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2228841008065262,0.12394816162749331,4.410320546352619>, <-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 0.5 }
    cylinder { m*<2.5567685454178672,0.011679856843626102,-1.793167673014278>, <-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 0.5}
    cylinder { m*<-1.79955520848128,2.2381198258758506,-1.5379039129790646>, <-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 0.5 }
    cylinder {  m*<-1.5317679874434482,-2.649572116528047,-1.3483576278164922>, <-0.17793984858839001,-0.09035411854274808,-0.563958147563096>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    