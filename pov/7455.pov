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
    sphere { m*<-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 1 }        
    sphere {  m*<0.8246705265744417,0.17470036625489116,9.324421947310709>, 1 }
    sphere {  m*<8.192457724897242,-0.11039188453737103,-5.246255481763223>, 1 }
    sphere {  m*<-6.7035054687917475,6.41268948908327,-3.7554485785816167>, 1}
    sphere { m*<-3.131429761722683,-6.340182817213138,-1.6996901402618314>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8246705265744417,0.17470036625489116,9.324421947310709>, <-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 0.5 }
    cylinder { m*<8.192457724897242,-0.11039188453737103,-5.246255481763223>, <-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 0.5}
    cylinder { m*<-6.7035054687917475,6.41268948908327,-3.7554485785816167>, <-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 0.5 }
    cylinder {  m*<-3.131429761722683,-6.340182817213138,-1.6996901402618314>, <-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 0.5}

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
    sphere { m*<-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 1 }        
    sphere {  m*<0.8246705265744417,0.17470036625489116,9.324421947310709>, 1 }
    sphere {  m*<8.192457724897242,-0.11039188453737103,-5.246255481763223>, 1 }
    sphere {  m*<-6.7035054687917475,6.41268948908327,-3.7554485785816167>, 1}
    sphere { m*<-3.131429761722683,-6.340182817213138,-1.6996901402618314>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8246705265744417,0.17470036625489116,9.324421947310709>, <-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 0.5 }
    cylinder { m*<8.192457724897242,-0.11039188453737103,-5.246255481763223>, <-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 0.5}
    cylinder { m*<-6.7035054687917475,6.41268948908327,-3.7554485785816167>, <-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 0.5 }
    cylinder {  m*<-3.131429761722683,-6.340182817213138,-1.6996901402618314>, <-0.5944969676257207,-0.8152385476250266,-0.5248681497244428>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    