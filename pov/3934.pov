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
    sphere { m*<-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 1 }        
    sphere {  m*<0.12328294767675602,0.11456201689334722,2.7914532753531587>, 1 }
    sphere {  m*<2.6172562369413273,0.08788591409939639,-1.425311021218576>, 1 }
    sphere {  m*<-1.7390675169578267,2.3143258831316245,-1.1700472611833619>, 1}
    sphere { m*<-1.6766108969796676,-2.9615141223000325,-1.0994682305258765>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12328294767675602,0.11456201689334722,2.7914532753531587>, <-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 0.5 }
    cylinder { m*<2.6172562369413273,0.08788591409939639,-1.425311021218576>, <-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 0.5}
    cylinder { m*<-1.7390675169578267,2.3143258831316245,-1.1700472611833619>, <-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 0.5 }
    cylinder {  m*<-1.6766108969796676,-2.9615141223000325,-1.0994682305258765>, <-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 0.5}

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
    sphere { m*<-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 1 }        
    sphere {  m*<0.12328294767675602,0.11456201689334722,2.7914532753531587>, 1 }
    sphere {  m*<2.6172562369413273,0.08788591409939639,-1.425311021218576>, 1 }
    sphere {  m*<-1.7390675169578267,2.3143258831316245,-1.1700472611833619>, 1}
    sphere { m*<-1.6766108969796676,-2.9615141223000325,-1.0994682305258765>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12328294767675602,0.11456201689334722,2.7914532753531587>, <-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 0.5 }
    cylinder { m*<2.6172562369413273,0.08788591409939639,-1.425311021218576>, <-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 0.5}
    cylinder { m*<-1.7390675169578267,2.3143258831316245,-1.1700472611833619>, <-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 0.5 }
    cylinder {  m*<-1.6766108969796676,-2.9615141223000325,-1.0994682305258765>, <-0.1174521570649355,-0.014148061286977986,-0.196101495767391>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    