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
    sphere { m*<-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 1 }        
    sphere {  m*<0.24966022456770204,0.28473055412425535,8.529740165447127>, 1 }
    sphere {  m*<4.968626184307033,0.04699649213716081,-4.290881175918857>, 1 }
    sphere {  m*<-2.5606036933532823,2.166520718092112,-2.312928655765461>, 1}
    sphere { m*<-2.292816472315451,-2.721171224311785,-2.1233823706028905>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24966022456770204,0.28473055412425535,8.529740165447127>, <-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 0.5 }
    cylinder { m*<4.968626184307033,0.04699649213716081,-4.290881175918857>, <-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 0.5}
    cylinder { m*<-2.5606036933532823,2.166520718092112,-2.312928655765461>, <-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 0.5 }
    cylinder {  m*<-2.292816472315451,-2.721171224311785,-2.1233823706028905>, <-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 0.5}

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
    sphere { m*<-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 1 }        
    sphere {  m*<0.24966022456770204,0.28473055412425535,8.529740165447127>, 1 }
    sphere {  m*<4.968626184307033,0.04699649213716081,-4.290881175918857>, 1 }
    sphere {  m*<-2.5606036933532823,2.166520718092112,-2.312928655765461>, 1}
    sphere { m*<-2.292816472315451,-2.721171224311785,-2.1233823706028905>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24966022456770204,0.28473055412425535,8.529740165447127>, <-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 0.5 }
    cylinder { m*<4.968626184307033,0.04699649213716081,-4.290881175918857>, <-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 0.5}
    cylinder { m*<-2.5606036933532823,2.166520718092112,-2.312928655765461>, <-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 0.5 }
    cylinder {  m*<-2.292816472315451,-2.721171224311785,-2.1233823706028905>, <-0.9083886302760742,-0.162364432419139,-1.3928897990132971>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    