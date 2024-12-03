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
    sphere { m*<-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 1 }        
    sphere {  m*<-0.02191893913632703,0.1549497379943155,8.909486720810555>, 1 }
    sphere {  m*<7.333432498863647,0.0660294619999583,-5.670006569234802>, 1 }
    sphere {  m*<-3.9019620086911253,2.86100225735774,-2.2101014240681103>, 1}
    sphere { m*<-2.84608929778534,-2.9579623112208506,-1.6418313124451545>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.02191893913632703,0.1549497379943155,8.909486720810555>, <-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 0.5 }
    cylinder { m*<7.333432498863647,0.0660294619999583,-5.670006569234802>, <-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 0.5}
    cylinder { m*<-3.9019620086911253,2.86100225735774,-2.2101014240681103>, <-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 0.5 }
    cylinder {  m*<-2.84608929778534,-2.9579623112208506,-1.6418313124451545>, <-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 0.5}

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
    sphere { m*<-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 1 }        
    sphere {  m*<-0.02191893913632703,0.1549497379943155,8.909486720810555>, 1 }
    sphere {  m*<7.333432498863647,0.0660294619999583,-5.670006569234802>, 1 }
    sphere {  m*<-3.9019620086911253,2.86100225735774,-2.2101014240681103>, 1}
    sphere { m*<-2.84608929778534,-2.9579623112208506,-1.6418313124451545>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.02191893913632703,0.1549497379943155,8.909486720810555>, <-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 0.5 }
    cylinder { m*<7.333432498863647,0.0660294619999583,-5.670006569234802>, <-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 0.5}
    cylinder { m*<-3.9019620086911253,2.86100225735774,-2.2101014240681103>, <-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 0.5 }
    cylinder {  m*<-2.84608929778534,-2.9579623112208506,-1.6418313124451545>, <-1.4819222631922468,-0.3723937571521348,-0.9693796977310385>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    