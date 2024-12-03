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
    sphere { m*<0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 1 }        
    sphere {  m*<0.6496093487698051,-2.6224065510496127e-18,3.9798900316611086>, 1 }
    sphere {  m*<7.205039475435342,1.2752826396297582e-18,-1.5786099748917672>, 1 }
    sphere {  m*<-4.2404851178248935,8.164965809277259,-2.2185315119023006>, 1}
    sphere { m*<-4.2404851178248935,-8.164965809277259,-2.218531511902304>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6496093487698051,-2.6224065510496127e-18,3.9798900316611086>, <0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 0.5 }
    cylinder { m*<7.205039475435342,1.2752826396297582e-18,-1.5786099748917672>, <0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 0.5}
    cylinder { m*<-4.2404851178248935,8.164965809277259,-2.2185315119023006>, <0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 0.5 }
    cylinder {  m*<-4.2404851178248935,-8.164965809277259,-2.218531511902304>, <0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 0.5}

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
    sphere { m*<0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 1 }        
    sphere {  m*<0.6496093487698051,-2.6224065510496127e-18,3.9798900316611086>, 1 }
    sphere {  m*<7.205039475435342,1.2752826396297582e-18,-1.5786099748917672>, 1 }
    sphere {  m*<-4.2404851178248935,8.164965809277259,-2.2185315119023006>, 1}
    sphere { m*<-4.2404851178248935,-8.164965809277259,-2.218531511902304>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6496093487698051,-2.6224065510496127e-18,3.9798900316611086>, <0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 0.5 }
    cylinder { m*<7.205039475435342,1.2752826396297582e-18,-1.5786099748917672>, <0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 0.5}
    cylinder { m*<-4.2404851178248935,8.164965809277259,-2.2185315119023006>, <0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 0.5 }
    cylinder {  m*<-4.2404851178248935,-8.164965809277259,-2.218531511902304>, <0.5653385988793813,-6.3640105856067576e-18,0.9810709650224486>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    