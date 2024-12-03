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
    sphere { m*<1.0328436142124524,0.4191824358420433,0.47655314323714504>, 1 }        
    sphere {  m*<1.2768123418655464,0.45268460152082524,3.466427455703463>, 1 }
    sphere {  m*<3.7700595309280818,0.4526846015208251,-0.750854752787155>, 1 }
    sphere {  m*<-2.9357138288916516,6.678550853690527,-1.8699137839706805>, 1}
    sphere { m*<-3.799635432955592,-7.855925283018293,-2.3800463114732597>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2768123418655464,0.45268460152082524,3.466427455703463>, <1.0328436142124524,0.4191824358420433,0.47655314323714504>, 0.5 }
    cylinder { m*<3.7700595309280818,0.4526846015208251,-0.750854752787155>, <1.0328436142124524,0.4191824358420433,0.47655314323714504>, 0.5}
    cylinder { m*<-2.9357138288916516,6.678550853690527,-1.8699137839706805>, <1.0328436142124524,0.4191824358420433,0.47655314323714504>, 0.5 }
    cylinder {  m*<-3.799635432955592,-7.855925283018293,-2.3800463114732597>, <1.0328436142124524,0.4191824358420433,0.47655314323714504>, 0.5}

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
    sphere { m*<1.0328436142124524,0.4191824358420433,0.47655314323714504>, 1 }        
    sphere {  m*<1.2768123418655464,0.45268460152082524,3.466427455703463>, 1 }
    sphere {  m*<3.7700595309280818,0.4526846015208251,-0.750854752787155>, 1 }
    sphere {  m*<-2.9357138288916516,6.678550853690527,-1.8699137839706805>, 1}
    sphere { m*<-3.799635432955592,-7.855925283018293,-2.3800463114732597>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2768123418655464,0.45268460152082524,3.466427455703463>, <1.0328436142124524,0.4191824358420433,0.47655314323714504>, 0.5 }
    cylinder { m*<3.7700595309280818,0.4526846015208251,-0.750854752787155>, <1.0328436142124524,0.4191824358420433,0.47655314323714504>, 0.5}
    cylinder { m*<-2.9357138288916516,6.678550853690527,-1.8699137839706805>, <1.0328436142124524,0.4191824358420433,0.47655314323714504>, 0.5 }
    cylinder {  m*<-3.799635432955592,-7.855925283018293,-2.3800463114732597>, <1.0328436142124524,0.4191824358420433,0.47655314323714504>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    