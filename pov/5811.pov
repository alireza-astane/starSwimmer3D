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
    sphere { m*<-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 1 }        
    sphere {  m*<0.028472756379153252,0.2801065034326593,8.722984231625878>, 1 }
    sphere {  m*<6.378884782621818,0.09022583730167014,-5.1675239198045>, 1 }
    sphere {  m*<-2.996315200389154,2.152338013863448,-2.063456111017362>, 1}
    sphere { m*<-2.728527979351323,-2.7353539285404493,-1.8739098258547915>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.028472756379153252,0.2801065034326593,8.722984231625878>, <-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 0.5 }
    cylinder { m*<6.378884782621818,0.09022583730167014,-5.1675239198045>, <-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 0.5}
    cylinder { m*<-2.996315200389154,2.152338013863448,-2.063456111017362>, <-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 0.5 }
    cylinder {  m*<-2.728527979351323,-2.7353539285404493,-1.8739098258547915>, <-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 0.5}

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
    sphere { m*<-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 1 }        
    sphere {  m*<0.028472756379153252,0.2801065034326593,8.722984231625878>, 1 }
    sphere {  m*<6.378884782621818,0.09022583730167014,-5.1675239198045>, 1 }
    sphere {  m*<-2.996315200389154,2.152338013863448,-2.063456111017362>, 1}
    sphere { m*<-2.728527979351323,-2.7353539285404493,-1.8739098258547915>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.028472756379153252,0.2801065034326593,8.722984231625878>, <-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 0.5 }
    cylinder { m*<6.378884782621818,0.09022583730167014,-5.1675239198045>, <-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 0.5}
    cylinder { m*<-2.996315200389154,2.152338013863448,-2.063456111017362>, <-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 0.5 }
    cylinder {  m*<-2.728527979351323,-2.7353539285404493,-1.8739098258547915>, <-1.327832317665223,-0.17684135352298838,-1.174030874897619>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    