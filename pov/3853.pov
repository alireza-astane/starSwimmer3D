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
    sphere { m*<-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 1 }        
    sphere {  m*<0.16704019839293904,0.19727882588943868,2.8168059502858274>, 1 }
    sphere {  m*<2.66101348765751,0.17060272309548763,-1.399958346285909>, 1 }
    sphere {  m*<-1.695310266241644,2.397042692127716,-1.144694586250695>, 1}
    sphere { m*<-1.8887271645070909,-3.362489514923162,-1.2223670573611023>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16704019839293904,0.19727882588943868,2.8168059502858274>, <-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 0.5 }
    cylinder { m*<2.66101348765751,0.17060272309548763,-1.399958346285909>, <-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 0.5}
    cylinder { m*<-1.695310266241644,2.397042692127716,-1.144694586250695>, <-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 0.5 }
    cylinder {  m*<-1.8887271645070909,-3.362489514923162,-1.2223670573611023>, <-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 0.5}

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
    sphere { m*<-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 1 }        
    sphere {  m*<0.16704019839293904,0.19727882588943868,2.8168059502858274>, 1 }
    sphere {  m*<2.66101348765751,0.17060272309548763,-1.399958346285909>, 1 }
    sphere {  m*<-1.695310266241644,2.397042692127716,-1.144694586250695>, 1}
    sphere { m*<-1.8887271645070909,-3.362489514923162,-1.2223670573611023>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16704019839293904,0.19727882588943868,2.8168059502858274>, <-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 0.5 }
    cylinder { m*<2.66101348765751,0.17060272309548763,-1.399958346285909>, <-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 0.5}
    cylinder { m*<-1.695310266241644,2.397042692127716,-1.144694586250695>, <-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 0.5 }
    cylinder {  m*<-1.8887271645070909,-3.362489514923162,-1.2223670573611023>, <-0.07369490634875242,0.06856874770911348,-0.17074882083472293>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    