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
    sphere { m*<-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 1 }        
    sphere {  m*<-0.1251834615899228,0.2768273264444162,8.855498555670907>, 1 }
    sphere {  m*<7.232244112987935,0.11544275083071409,-5.722332491792928>, 1 }
    sphere {  m*<-3.2683526047395737,2.1438604719708296,-1.8979375744350508>, 1}
    sphere { m*<-3.0005653837017423,-2.7438314704330677,-1.7083912892724804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.1251834615899228,0.2768273264444162,8.855498555670907>, <-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 0.5 }
    cylinder { m*<7.232244112987935,0.11544275083071409,-5.722332491792928>, <-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 0.5}
    cylinder { m*<-3.2683526047395737,2.1438604719708296,-1.8979375744350508>, <-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 0.5 }
    cylinder {  m*<-3.0005653837017423,-2.7438314704330677,-1.7083912892724804>, <-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 0.5}

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
    sphere { m*<-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 1 }        
    sphere {  m*<-0.1251834615899228,0.2768273264444162,8.855498555670907>, 1 }
    sphere {  m*<7.232244112987935,0.11544275083071409,-5.722332491792928>, 1 }
    sphere {  m*<-3.2683526047395737,2.1438604719708296,-1.8979375744350508>, 1}
    sphere { m*<-3.0005653837017423,-2.7438314704330677,-1.7083912892724804>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.1251834615899228,0.2768273264444162,8.855498555670907>, <-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 0.5 }
    cylinder { m*<7.232244112987935,0.11544275083071409,-5.722332491792928>, <-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 0.5}
    cylinder { m*<-3.2683526047395737,2.1438604719708296,-1.8979375744350508>, <-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 0.5 }
    cylinder {  m*<-3.0005653837017423,-2.7438314704330677,-1.7083912892724804>, <-1.5911195833946554,-0.18550172305163093,-1.0256142049511152>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    