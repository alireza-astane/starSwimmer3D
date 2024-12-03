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
    sphere { m*<0.17417593462224595,0.537133035289374,-0.02713401977729088>, 1 }        
    sphere {  m*<0.4149110393639376,0.6658431134696995,2.9604207513432597>, 1 }
    sphere {  m*<2.908884328628503,0.6391670106757483,-1.256343545228476>, 1 }
    sphere {  m*<-1.4474394252706442,2.865606979707975,-1.0010797851932614>, 1}
    sphere { m*<-2.8759988622181973,-5.228785114171491,-1.7943860487881849>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4149110393639376,0.6658431134696995,2.9604207513432597>, <0.17417593462224595,0.537133035289374,-0.02713401977729088>, 0.5 }
    cylinder { m*<2.908884328628503,0.6391670106757483,-1.256343545228476>, <0.17417593462224595,0.537133035289374,-0.02713401977729088>, 0.5}
    cylinder { m*<-1.4474394252706442,2.865606979707975,-1.0010797851932614>, <0.17417593462224595,0.537133035289374,-0.02713401977729088>, 0.5 }
    cylinder {  m*<-2.8759988622181973,-5.228785114171491,-1.7943860487881849>, <0.17417593462224595,0.537133035289374,-0.02713401977729088>, 0.5}

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
    sphere { m*<0.17417593462224595,0.537133035289374,-0.02713401977729088>, 1 }        
    sphere {  m*<0.4149110393639376,0.6658431134696995,2.9604207513432597>, 1 }
    sphere {  m*<2.908884328628503,0.6391670106757483,-1.256343545228476>, 1 }
    sphere {  m*<-1.4474394252706442,2.865606979707975,-1.0010797851932614>, 1}
    sphere { m*<-2.8759988622181973,-5.228785114171491,-1.7943860487881849>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4149110393639376,0.6658431134696995,2.9604207513432597>, <0.17417593462224595,0.537133035289374,-0.02713401977729088>, 0.5 }
    cylinder { m*<2.908884328628503,0.6391670106757483,-1.256343545228476>, <0.17417593462224595,0.537133035289374,-0.02713401977729088>, 0.5}
    cylinder { m*<-1.4474394252706442,2.865606979707975,-1.0010797851932614>, <0.17417593462224595,0.537133035289374,-0.02713401977729088>, 0.5 }
    cylinder {  m*<-2.8759988622181973,-5.228785114171491,-1.7943860487881849>, <0.17417593462224595,0.537133035289374,-0.02713401977729088>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    