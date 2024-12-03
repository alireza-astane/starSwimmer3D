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
    sphere { m*<0.7365994857015659,0.8721511952170633,0.30139512708330735>, 1 }        
    sphere {  m*<0.9792974216369901,0.9511071558965677,3.29051622451363>, 1 }
    sphere {  m*<3.472544610699525,0.9511071558965674,-0.9267659839769846>, 1 }
    sphere {  m*<-1.973708889069409,4.919699697776645,-1.3011042293118766>, 1}
    sphere { m*<-3.9072085389601505,-7.547516288740415,-2.443656343457529>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9792974216369901,0.9511071558965677,3.29051622451363>, <0.7365994857015659,0.8721511952170633,0.30139512708330735>, 0.5 }
    cylinder { m*<3.472544610699525,0.9511071558965674,-0.9267659839769846>, <0.7365994857015659,0.8721511952170633,0.30139512708330735>, 0.5}
    cylinder { m*<-1.973708889069409,4.919699697776645,-1.3011042293118766>, <0.7365994857015659,0.8721511952170633,0.30139512708330735>, 0.5 }
    cylinder {  m*<-3.9072085389601505,-7.547516288740415,-2.443656343457529>, <0.7365994857015659,0.8721511952170633,0.30139512708330735>, 0.5}

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
    sphere { m*<0.7365994857015659,0.8721511952170633,0.30139512708330735>, 1 }        
    sphere {  m*<0.9792974216369901,0.9511071558965677,3.29051622451363>, 1 }
    sphere {  m*<3.472544610699525,0.9511071558965674,-0.9267659839769846>, 1 }
    sphere {  m*<-1.973708889069409,4.919699697776645,-1.3011042293118766>, 1}
    sphere { m*<-3.9072085389601505,-7.547516288740415,-2.443656343457529>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9792974216369901,0.9511071558965677,3.29051622451363>, <0.7365994857015659,0.8721511952170633,0.30139512708330735>, 0.5 }
    cylinder { m*<3.472544610699525,0.9511071558965674,-0.9267659839769846>, <0.7365994857015659,0.8721511952170633,0.30139512708330735>, 0.5}
    cylinder { m*<-1.973708889069409,4.919699697776645,-1.3011042293118766>, <0.7365994857015659,0.8721511952170633,0.30139512708330735>, 0.5 }
    cylinder {  m*<-3.9072085389601505,-7.547516288740415,-2.443656343457529>, <0.7365994857015659,0.8721511952170633,0.30139512708330735>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    